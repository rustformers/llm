import { Params } from "./hooks/useStore";
export const instructionString = "{{instruction}}";

export const defaultPrompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
${instructionString}

### Response:`;

export const defaultParams: Params = {
  n_batch: 8,
  top_k: 40,
  top_p: 0.95,
  repeat_penalty: 1.3,
  temp: 0.8,
  num_predict: 512,
};
