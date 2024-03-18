from langchain_community.llms import VLLM
from typing import Any, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult

class CustVLLM(VLLM):

    def _generate(
        self,
        prompts: List[str],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input.
        Diff -> VLLM: modify to propery inherit from VLLM's attribute."""

        from vllm import SamplingParams

        params = {**self._default_params, **kwargs, "stop": self.stop}
        sampling_params = SamplingParams(**params)

        outputs = self.client.generate(prompts, sampling_params)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)