"""Uses API calls to generate explanations of neuron behavior."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Union

import numpy as np

from neuron_explainer.activations.activation_records import (
    calculate_max_activation,
    format_activation_records,
    non_zero_activation_proportion,
)
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.activations.attention_utils import (
    convert_flattened_index_to_unflattened_index,
)
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.few_shot_examples import (
    ATTENTION_HEAD_FEW_SHOT_EXAMPLES,
    AttentionTokenPairExample,
    FewShotExampleSet,
)
from neuron_explainer.explanations.prompt_builder import (
    HarmonyMessage,
    PromptBuilder,
    PromptFormat,
    Role,
)
from neuron_explainer.explanations.token_space_few_shot_examples import (
    TokenSpaceFewShotExampleSet,
)

logger = logging.getLogger(__name__)
ATTENTION_EXPLANATION_PREFIX = "this attention head"
ATTENTION_SEQUENCE_SEPARATOR = "<|sequence_separator|>"


# TODO(williamrs): This prefix may not work well for some things, like predicting the next token.
# Try other options like "this neuron activates for".
EXPLANATION_PREFIX = "the main thing this neuron does is find"


def _split_numbered_list(text: str) -> list[str]:
    """Split a numbered list into a list of strings."""
    lines = re.split(r"\n\d+\.", text)
    # Strip the leading whitespace from each line.
    return [line.lstrip() for line in lines]


def _remove_final_period(text: str) -> str:
    """Strip a final period or period-space from a string."""
    if text.endswith("."):
        return text[:-1]
    elif text.endswith(". "):
        return text[:-2]
    return text


# TODO: should pull from API and/or combine with the HARMONY_V4_MODELS
class ContextSize(int, Enum):
    TWO_K = 2049
    FOUR_K = 4097
    SIXTEEN_K = 16384
    ONETWENTYEIGHT_K = 128000

    @classmethod
    def from_int(cls, i: int) -> ContextSize:
        for context_size in cls:
            if context_size.value == i:
                return context_size
        raise ValueError(f"{i} is not a valid ContextSize")


# TODO: should pull these from API
HARMONY_V4_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4o-2024-05-13",
    "gpt-4-1106-preview",
    "gpt-4-turbo-2024-04-09",
]


class NeuronExplainer(ABC):
    """
    Abstract base class for Explainer classes that generate explanations from subclass-specific
    input data.
    """

    def __init__(
        self,
        model_name: str,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        # This parameter lets us adjust the length of the prompt when we're generating explanations
        # using older models with shorter context windows. In the future we can use it to experiment
        # with longer context windows.
        context_size: ContextSize = ContextSize.ONETWENTYEIGHT_K,
        max_concurrent: Optional[int] = 10,
        cache: bool = False,
    ):
        # if prompt_format == PromptFormat.HARMONY_V4:
        #     assert model_name in HARMONY_V4_MODELS
        if prompt_format in [PromptFormat.NONE, PromptFormat.INSTRUCTION_FOLLOWING]:
            assert model_name not in HARMONY_V4_MODELS
        # else:
        #     raise ValueError(f"Unhandled prompt format {prompt_format}")

        self.model_name = model_name
        self.prompt_format = prompt_format
        self.context_size = context_size
        self.client = ApiClient(
            model_name=model_name, max_concurrent=max_concurrent, cache=cache
        )

    async def generate_explanations(
        self,
        *,
        num_samples: int = 5,
        max_tokens: int = 60,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **prompt_kwargs: Any,
    ) -> list[Any]:
        """Generate explanations based on subclass-specific input data."""
        prompt = self.make_explanation_prompt(
            max_tokens_for_completion=max_tokens, **prompt_kwargs
        )

        generate_kwargs: dict[str, Any] = {
            "n": num_samples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if self.prompt_format == PromptFormat.HARMONY_V4:
            assert isinstance(prompt, list)
            assert isinstance(prompt[0], dict)  # Really a HarmonyMessage
            generate_kwargs["messages"] = prompt
        else:
            assert isinstance(prompt, str)
            generate_kwargs["prompt"] = prompt

        response = await self.client.make_request(**generate_kwargs)
        logger.debug("response in generate_explanations is %s", response)

        if self.prompt_format == PromptFormat.HARMONY_V4:
            explanations = [x["message"]["content"] for x in response["choices"]]
        elif self.prompt_format in [
            PromptFormat.NONE,
            PromptFormat.INSTRUCTION_FOLLOWING,
        ]:
            explanations = [x["text"] for x in response["choices"]]
        else:
            raise ValueError(f"Unhandled prompt format {self.prompt_format}")

        return self.postprocess_explanations(explanations, prompt_kwargs)

    @abstractmethod
    def make_explanation_prompt(
        self, **kwargs: Any
    ) -> Union[str, list[HarmonyMessage]]:
        """
        Create a prompt to send to the API to generate one or more explanations.

        A prompt can be a simple string, or a list of HarmonyMessages, depending on the PromptFormat
        used by this instance.
        """
        ...

    def postprocess_explanations(
        self, completions: list[str], prompt_kwargs: dict[str, Any]
    ) -> list[Any]:
        """Postprocess the completions returned by the API into a list of explanations."""
        return completions  # no-op by default

    def _prompt_is_too_long(
        self, prompt_builder: PromptBuilder, max_tokens_for_completion: int
    ) -> bool:
        # We'll get a context size error if the prompt itself plus the maximum number of tokens for
        # the completion is longer than the context size.
        prompt_length = prompt_builder.prompt_length_in_tokens(self.prompt_format)
        if prompt_length + max_tokens_for_completion > self.context_size.value:
            print(
                f"Prompt is too long: {prompt_length} + {max_tokens_for_completion} > "
                f"{self.context_size.value}"
            )
            return True
        return False


class TokenActivationPairExplainer(NeuronExplainer):
    """
    Generate explanations of neuron behavior using a prompt with lists of token/activation pairs.
    """

    def __init__(
        self,
        model_name: str,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        # This parameter lets us adjust the length of the prompt when we're generating explanations
        # using older models with shorter context windows. In the future we can use it to experiment
        # with 8k+ context windows.
        context_size: ContextSize = ContextSize.ONETWENTYEIGHT_K,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.ORIGINAL,
        repeat_non_zero_activations: bool = True,
        max_concurrent: Optional[int] = 10,
        cache: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            prompt_format=prompt_format,
            max_concurrent=max_concurrent,
            cache=cache,
        )
        self.context_size = context_size
        self.few_shot_example_set = few_shot_example_set
        self.repeat_non_zero_activations = repeat_non_zero_activations

    def make_explanation_prompt(
        self, **kwargs: Any
    ) -> Union[str, list[HarmonyMessage]]:
        original_kwargs = kwargs.copy()
        all_activation_records: Sequence[ActivationRecord] = kwargs.pop(
            "all_activation_records"
        )
        max_activation: float = kwargs.pop("max_activation")
        kwargs.setdefault("numbered_list_of_n_explanations", None)
        numbered_list_of_n_explanations: Optional[int] = kwargs.pop(
            "numbered_list_of_n_explanations"
        )
        if numbered_list_of_n_explanations is not None:
            assert numbered_list_of_n_explanations > 0, numbered_list_of_n_explanations
        # This parameter lets us dynamically shrink the prompt if our initial attempt to create it
        # results in something that's too long. It's only implemented for the 4k context size.
        kwargs.setdefault("omit_n_activation_records", 0)
        omit_n_activation_records: int = kwargs.pop("omit_n_activation_records")
        max_tokens_for_completion: int = kwargs.pop("max_tokens_for_completion")
        assert not kwargs, f"Unexpected kwargs: {kwargs}"

        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            "We're studying neurons in a neural network. Each neuron looks for some particular "
            "thing in a short document. Look at the parts of the document the neuron activates for "
            "and summarize in a single sentence what the neuron is looking for. Don't list "
            "examples of words.\n\nThe activation format is token<tab>activation. Activation "
            "values range from 0 to 10. A neuron finding what it's looking for is represented by a "
            "non-zero activation value. The higher the activation value, the stronger the match.",
        )
        few_shot_examples = self.few_shot_example_set.get_examples()
        num_omitted_activation_records = 0
        for i, few_shot_example in enumerate(few_shot_examples):
            few_shot_activation_records = few_shot_example.activation_records
            if self.context_size == ContextSize.TWO_K:
                # If we're using a 2k context window, we only have room for one activation record
                # per few-shot example. (Two few-shot examples with one activation record each seems
                # to work better than one few-shot example with two activation records, in local
                # testing.)
                few_shot_activation_records = few_shot_activation_records[:1]
            elif (
                self.context_size == ContextSize.FOUR_K
                and num_omitted_activation_records < omit_n_activation_records
            ):
                # Drop the last activation record for this few-shot example to save tokens, assuming
                # there are at least two activation records.
                if len(few_shot_activation_records) > 1:
                    print(
                        f"Warning: omitting activation record from few-shot example {i}"
                    )
                    few_shot_activation_records = few_shot_activation_records[:-1]
                    num_omitted_activation_records += 1
            self._add_per_neuron_explanation_prompt(
                prompt_builder,
                few_shot_activation_records,
                i,
                calculate_max_activation(few_shot_example.activation_records),
                numbered_list_of_n_explanations=numbered_list_of_n_explanations,
                explanation=few_shot_example.explanation,
            )
        self._add_per_neuron_explanation_prompt(
            prompt_builder,
            # If we're using a 2k context window, we only have room for two of the activation
            # records.
            (
                all_activation_records[:2]
                if self.context_size == ContextSize.TWO_K
                else all_activation_records
            ),
            len(few_shot_examples),
            max_activation,
            numbered_list_of_n_explanations=numbered_list_of_n_explanations,
            explanation=None,
        )
        # If the prompt is too long *and* we omitted the specified number of activation records, try
        # again, omitting one more. (If we didn't make the specified number of omissions, we're out
        # of opportunities to omit records, so we just return the prompt as-is.)
        if (
            self._prompt_is_too_long(prompt_builder, max_tokens_for_completion)
            and num_omitted_activation_records == omit_n_activation_records
        ):
            original_kwargs["omit_n_activation_records"] = omit_n_activation_records + 1
            return self.make_explanation_prompt(**original_kwargs)
        return prompt_builder.build(self.prompt_format)

    def _add_per_neuron_explanation_prompt(
        self,
        prompt_builder: PromptBuilder,
        activation_records: Sequence[ActivationRecord],
        index: int,
        max_activation: float,
        # When set, this indicates that the prompt should solicit a numbered list of the given
        # number of explanations, rather than a single explanation.
        numbered_list_of_n_explanations: Optional[int],
        explanation: Optional[str],  # None means this is the end of the full prompt.
    ) -> None:
        max_activation = calculate_max_activation(activation_records)
        user_message = f"""

Neuron {index + 1}
Activations:{format_activation_records(activation_records, max_activation, omit_zeros=False)}"""
        # We repeat the non-zero activations only if it was requested and if the proportion of
        # non-zero activations isn't too high.
        if (
            self.repeat_non_zero_activations
            and non_zero_activation_proportion(activation_records, max_activation) < 0.2
        ):
            user_message += (
                f"\nSame activations, but with all zeros filtered out:"
                f"{format_activation_records(activation_records, max_activation, omit_zeros=True)}"
            )

        if numbered_list_of_n_explanations is None:
            user_message += f"\nExplanation of neuron {index + 1} behavior:"
            assistant_message = ""
            # For the IF format, we want <|endofprompt|> to come before the explanation prefix.
            if self.prompt_format == PromptFormat.INSTRUCTION_FOLLOWING:
                assistant_message += f" {EXPLANATION_PREFIX}"
            else:
                user_message += f" {EXPLANATION_PREFIX}"
            prompt_builder.add_message(Role.USER, user_message)

            if explanation is not None:
                assistant_message += f" {explanation}."
            if assistant_message:
                prompt_builder.add_message(Role.ASSISTANT, assistant_message)
        else:
            if explanation is None:
                # For the final neuron, we solicit a numbered list of explanations.
                prompt_builder.add_message(
                    Role.USER,
                    f"""\nHere are {numbered_list_of_n_explanations} possible explanations for neuron {index + 1} behavior, each beginning with "{EXPLANATION_PREFIX}":\n1. {EXPLANATION_PREFIX}""",
                )
            else:
                # For the few-shot examples, we only present one explanation, but we present it as a
                # numbered list.
                prompt_builder.add_message(
                    Role.USER,
                    f"""\nHere is 1 possible explanation for neuron {index + 1} behavior, beginning with "{EXPLANATION_PREFIX}":\n1. {EXPLANATION_PREFIX}""",
                )
                prompt_builder.add_message(Role.ASSISTANT, f" {explanation}.")

    def postprocess_explanations(
        self, completions: list[str], prompt_kwargs: dict[str, Any]
    ) -> list[Any]:
        """Postprocess the explanations returned by the API"""
        numbered_list_of_n_explanations = prompt_kwargs.get(
            "numbered_list_of_n_explanations"
        )
        if numbered_list_of_n_explanations is None:
            return completions
        else:
            all_explanations = []
            for completion in completions:
                for explanation in _split_numbered_list(completion):
                    if explanation.startswith(EXPLANATION_PREFIX):
                        explanation = explanation[len(EXPLANATION_PREFIX) :]
                    all_explanations.append(explanation.strip())
            return all_explanations


class TokenSpaceRepresentationExplainer(NeuronExplainer):
    """
    Generate explanations of arbitrary lists of tokens which disproportionately activate a
    particular neuron. These lists of tokens can be generated in various ways. As an example, in one
    set of experiments, we compute the average activation for each neuron conditional on each token
    that appears in an internet text corpus. We then sort the tokens by their average activation,
    and show 50 of the top 100 tokens. Other techniques that could be used include taking the top
    tokens in the logit lens or tuned lens representations of a neuron.
    """

    def __init__(
        self,
        model_name: str,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        context_size: ContextSize = ContextSize.FOUR_K,
        few_shot_example_set: TokenSpaceFewShotExampleSet = TokenSpaceFewShotExampleSet.ORIGINAL,
        use_few_shot: bool = False,
        output_numbered_list: bool = False,
        max_concurrent: Optional[int] = 10,
        cache: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            prompt_format=prompt_format,
            context_size=context_size,
            max_concurrent=max_concurrent,
            cache=cache,
        )
        self.use_few_shot = use_few_shot
        self.output_numbered_list = output_numbered_list
        if self.use_few_shot:
            assert few_shot_example_set is not None
            self.few_shot_examples: Optional[TokenSpaceFewShotExampleSet] = (
                few_shot_example_set
            )
        else:
            self.few_shot_examples = None
        self.prompt_prefix = (
            "We're studying neurons in a neural network. Each neuron looks for some particular "
            "kind of token (which can be a word, or part of a word). Look at the tokens the neuron "
            "activates for (listed below) and summarize in a single sentence what the neuron is "
            "looking for. Don't list examples of words."
        )

    def make_explanation_prompt(
        self, **kwargs: Any
    ) -> Union[str, list[HarmonyMessage]]:
        tokens: list[str] = kwargs.pop("tokens")
        max_tokens_for_completion = kwargs.pop("max_tokens_for_completion")
        assert not kwargs, f"Unexpected kwargs: {kwargs}"
        # Note that this does not preserve the precise tokens, as e.g.
        # f" {token_with_no_leading_space}" may be tokenized as "f{token_with_leading_space}".
        # TODO(dan): Try out other variants, including "\n".join(...) and ",".join(...)
        stringified_tokens = ", ".join([f"'{t}'" for t in tokens])

        prompt_builder = PromptBuilder()
        prompt_builder.add_message(Role.SYSTEM, self.prompt_prefix)
        if self.use_few_shot:
            self._add_few_shot_examples(prompt_builder)
        self._add_neuron_specific_prompt(
            prompt_builder, stringified_tokens, explanation=None
        )

        if self._prompt_is_too_long(prompt_builder, max_tokens_for_completion):
            raise ValueError(
                f"Prompt too long: {prompt_builder.build(self.prompt_format)}"
            )
        else:
            return prompt_builder.build(self.prompt_format)

    def _add_few_shot_examples(self, prompt_builder: PromptBuilder) -> None:
        """
        Append few-shot examples to the prompt. Each one consists of a comma-delimited list of
        tokens and corresponding explanations, as saved in
        alignment/neuron_explainer/weight_explainer/token_space_few_shot_examples.py.
        """
        assert self.few_shot_examples is not None
        few_shot_example_list = self.few_shot_examples.get_examples()
        if self.output_numbered_list:
            raise NotImplementedError(
                "Numbered list output not supported for few-shot examples"
            )
        else:
            for few_shot_example in few_shot_example_list:
                self._add_neuron_specific_prompt(
                    prompt_builder,
                    ", ".join([f"'{t}'" for t in few_shot_example.tokens]),
                    explanation=few_shot_example.explanation,
                )

    def _add_neuron_specific_prompt(
        self,
        prompt_builder: PromptBuilder,
        stringified_tokens: str,
        explanation: Optional[str],
    ) -> None:
        """
        Append a neuron-specific prompt to the prompt builder. The prompt consists of a list of
        tokens followed by either an explanation (if one is passed, for few shot examples) or by
        the beginning of a completion, to be completed by the model with an explanation.
        """
        user_message = f"\n\n\n\nTokens:\n{stringified_tokens}\n\nExplanation:\n"
        assistant_message = ""
        looking_for = "This neuron is looking for"
        if self.prompt_format == PromptFormat.INSTRUCTION_FOLLOWING:
            # We want <|endofprompt|> to come before "This neuron is looking for" in the IF format.
            assistant_message += looking_for
        else:
            user_message += looking_for
        if self.output_numbered_list:
            start_of_list = "\n1."
            if self.prompt_format == PromptFormat.INSTRUCTION_FOLLOWING:
                assistant_message += start_of_list
            else:
                user_message += start_of_list
        if explanation is not None:
            assistant_message += f"{explanation}."
        prompt_builder.add_message(Role.USER, user_message)
        if assistant_message:
            prompt_builder.add_message(Role.ASSISTANT, assistant_message)

    def postprocess_explanations(
        self, completions: list[str], prompt_kwargs: dict[str, Any]
    ) -> list[str]:
        if self.output_numbered_list:
            # Each list in the top-level list will have multiple explanations (multiple strings).
            all_explanations = []
            for completion in completions:
                for explanation in _split_numbered_list(completion):
                    if explanation.startswith(EXPLANATION_PREFIX):
                        explanation = explanation[len(EXPLANATION_PREFIX) :]
                    all_explanations.append(explanation.strip())
            return all_explanations
        else:
            # Each element in the top-level list will be an explanation as a string.
            return [_remove_final_period(explanation) for explanation in completions]


def format_attention_head_token_pairs(
    token_pair_examples: list[AttentionTokenPairExample], omit_zeros: bool = False
) -> str:
    if omit_zeros:
        return ", ".join(
            [
                ", ".join(
                    [
                        f"({example.tokens[coords[1]]}, {example.tokens[coords[0]]})"
                        for coords in example.token_pair_coordinates
                    ]
                )
                for example in token_pair_examples
            ]
        )
    else:
        return f"\n{ATTENTION_SEQUENCE_SEPARATOR}\n".join(
            [
                f"\n{ATTENTION_SEQUENCE_SEPARATOR}\n".join(
                    [
                        f"{format_attention_head_token_pair_string(example.tokens, coords)}"
                        for coords in example.token_pair_coordinates
                    ]
                )
                for example in token_pair_examples
            ]
        )


def format_attention_head_token_pair_string(
    token_list: list[str], pair_coordinates: tuple[int, int]
) -> str:
    def format_activated_token(i: int, token: str) -> str:
        if i == pair_coordinates[0] and i == pair_coordinates[1]:
            return f"[[**{token}**]]"  # from and to
        if i == pair_coordinates[0]:
            return f"[[{token}]]"  # from
        if i == pair_coordinates[1]:
            return f"**{token}**"  # to
        return token

    return "".join(
        [format_activated_token(i, token) for i, token in enumerate(token_list)]
    )


def get_top_attention_coordinates(
    activation_records: list[ActivationRecord], top_k: int = 5
) -> list[tuple[int, float, tuple[int, int]]]:
    candidates = []
    for i, record in enumerate(activation_records):
        top_activation_flat_indices = np.argsort(record.activations)[::-1][:top_k]
        top_vals: list[float] = [
            record.activations[idx] for idx in top_activation_flat_indices
        ]
        top_coordinates = [
            convert_flattened_index_to_unflattened_index(flat_index)
            for flat_index in top_activation_flat_indices
        ]
        candidates.extend(
            [(i, top_val, coords) for top_val, coords in zip(top_vals, top_coordinates)]
        )
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]


class AttentionHeadExplainer(NeuronExplainer):
    """
    Generate explanations of attention head behavior using a prompt with lists of
    strongly attending to/from token pairs.
    Takes in NeuronRecord's corresponding to a single attention head. Extracts strongly
    activating to/from token pairs.
    """

    def __init__(
        self,
        model_name: str,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        # This parameter lets us adjust the length of the prompt when we're generating explanations
        # using older models with shorter context windows. In the future we can use it to experiment
        # with 8k+ context windows.
        context_size: ContextSize = ContextSize.ONETWENTYEIGHT_K,
        repeat_strongly_attending_pairs: bool = False,
        max_concurrent: int | None = 10,
        cache: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            prompt_format=prompt_format,
            max_concurrent=max_concurrent,
            cache=cache,
        )
        assert (
            context_size != ContextSize.TWO_K
        ), "2k context size not supported for attention explanation"
        self.context_size = context_size
        self.repeat_strongly_attending_pairs = repeat_strongly_attending_pairs

    def make_explanation_prompt(self, **kwargs: Any) -> str | list[HarmonyMessage]:
        original_kwargs = kwargs.copy()
        all_activation_records: list[ActivationRecord] = kwargs.pop(
            "all_activation_records"
        )
        # This parameter lets us dynamically shrink the prompt if our initial attempt to create it
        # results in something that's too long.
        kwargs.setdefault("omit_n_token_pair_examples", 0)
        omit_n_token_pair_examples: int = kwargs.pop("omit_n_token_pair_examples")

        max_tokens_for_completion: int = kwargs.pop("max_tokens_for_completion")

        kwargs.setdefault("num_top_pairs_to_display", 0)
        num_top_pairs_to_display: int = kwargs.pop("num_top_pairs_to_display")

        assert not kwargs, f"Unexpected kwargs: {kwargs}"

        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            "We're studying attention heads in a neural network. Each head looks at every pair of tokens "
            "in a short token sequence and activates for pairs of tokens that fit what it is looking for. "
            "Attention heads always attend from a token to a token earlier in the sequence (or from a "
            'token to itself). We will display multiple instances of sequences with the "to" token '
            'surrounded by double asterisks (e.g., **token**) and the "from" token surrounded by double '
            "square brackets (e.g., [[token]]). If a token attends from itself to itself, it will be "
            "surrounded by both (e.g., [[**token**]]). Look at the pairs of tokens the head activates for "
            "and summarize in a single sentence what pattern the head is looking for. We do not display "
            "every activating pair of tokens in a sequence; you must generalize from limited examples. "
            "Remember, the head always attends to tokens earlier in the sentence (marked with ** **) from "
            "tokens later in the sentence (marked with [[ ]]), except when the head attends from a token to "
            'itself (marked with [[** **]]). The explanation takes the form: "This attention head attends '
            "to {pattern of tokens marked with ** **, which appear earlier} from {pattern of tokens marked with "
            '[[ ]], which appear later}." The explanation does not include any of the markers (** **, [[ ]]), '
            f"as these are just for your reference. Sequences are separated by `{ATTENTION_SEQUENCE_SEPARATOR}`.",
        )
        num_omitted_token_pair_examples = 0
        for i, few_shot_example in enumerate(ATTENTION_HEAD_FEW_SHOT_EXAMPLES):
            few_shot_token_pair_examples = few_shot_example.token_pair_examples
            if num_omitted_token_pair_examples < omit_n_token_pair_examples:
                # Drop the last activation record for this few-shot example to save tokens, assuming
                # there are at least two activation records.
                if len(few_shot_token_pair_examples) > 1:
                    print(
                        f"Warning: omitting activation record from few-shot example {i}"
                    )
                    few_shot_token_pair_examples = few_shot_token_pair_examples[:-1]
                    num_omitted_token_pair_examples += 1
            few_shot_explanation: str = few_shot_example.explanation
            self._add_per_head_explanation_prompt(
                prompt_builder,
                few_shot_token_pair_examples,
                i,
                explanation=few_shot_explanation,
            )

        # ================================
        # Comment (Johnny): the original code does not seem to work (or I am not using it correctly?). Re-written below
        # ================================
        # # each element is (record_index, attention value, (from_token_index, to_token_index))
        # coords = get_top_attention_coordinates(
        #     all_activation_records, top_k=num_top_pairs_to_display
        # )
        # prompt_examples = {}
        # for record_index, _, (from_token_index, to_token_index) in coords:
        #     if record_index not in prompt_examples:
        #         prompt_examples[record_index] = AttentionTokenPairExample(
        #             tokens=all_activation_records[record_index].tokens,
        #             token_pair_coordinates=[(from_token_index, to_token_index)],
        #         )
        #     else:
        #         prompt_examples[record_index].token_pair_coordinates.append(
        #             (from_token_index, to_token_index)
        #         )
        # current_head_token_pair_examples = list(prompt_examples.values())

        # make list of attention token pair examples
        attention_token_pair_examples = []
        for i, activation_record in enumerate(all_activation_records):
            # from (first value) is the dfaTargetIndex
            from_index = activation_record.dfa_target_index
            # to (second value) is the index of the max dfa
            to_index = np.argmax(activation_record.dfa_values)
            attention_token_pair_examples.append(
                AttentionTokenPairExample(
                    tokens=activation_record.tokens,
                    token_pair_coordinates=[(from_index, to_index)],
                )
            )

        self._add_per_head_explanation_prompt(
            prompt_builder,
            attention_token_pair_examples,
            len(ATTENTION_HEAD_FEW_SHOT_EXAMPLES),
            explanation=None,
        )
        # If the prompt is too long *and* we omitted the specified number of activation records, try
        # again, omitting one more. (If we didn't make the specified number of omissions, we're out
        # of opportunities to omit records, so we just return the prompt as-is.)
        # if (
        #     self._prompt_is_too_long(prompt_builder, max_tokens_for_completion)
        #     and num_omitted_token_pair_examples == omit_n_token_pair_examples
        # ):
        #     original_kwargs["omit_n_token_pair_examples"] = (
        #         omit_n_token_pair_examples + 1
        #     )
        #     return self.make_explanation_prompt(**original_kwargs)
        return prompt_builder.build(self.prompt_format)

    def _add_per_head_explanation_prompt(
        self,
        prompt_builder: PromptBuilder,
        token_pair_examples: list[
            AttentionTokenPairExample
        ],  # each dict has keys "tokens" and "token_pair_coordinates"
        index: int,
        explanation: str | None,  # None means this is the end of the full prompt.
    ) -> None:
        user_message = f"""

Attention head {index + 1}
Activations:\n{format_attention_head_token_pairs(token_pair_examples, omit_zeros=False)}"""
        if self.repeat_strongly_attending_pairs:
            user_message += (
                f"\nThe same list of strongly activating token pairs, presented as (to_token, from_token):"
                f"{format_attention_head_token_pairs(token_pair_examples, omit_zeros=True)}"
            )

        user_message += f"\nExplanation of attention head {index + 1} behavior:"
        assistant_message = ""
        # For the IF format, we want <|endofprompt|> to come before the explanation prefix.
        if self.prompt_format == PromptFormat.INSTRUCTION_FOLLOWING:
            assistant_message += f" {ATTENTION_EXPLANATION_PREFIX}"
        else:
            user_message += f" {ATTENTION_EXPLANATION_PREFIX}"
        prompt_builder.add_message(Role.USER, user_message)

        if explanation is not None:
            assistant_message += f" {explanation}."
        if assistant_message:
            prompt_builder.add_message(Role.ASSISTANT, assistant_message)
