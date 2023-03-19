import { Params } from "./hooks/useStore";
export const alpacaName = "ggml-alpaca-7b-q4.bin";
export const alpacaUrl = "https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC";

export const instructionString = "{{instruction}}";

export const defaultPrompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
${instructionString}

### Response:
`;

export const defaultParams: Params = {
  n_batch: 8,
  top_k: 40,
  top_p: 0.95,
  repeat_penalty: 1.3,
  temp: 0.8,
  num_predict: 512,
};
