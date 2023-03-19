import { instructionString } from "../config";

export const getRandomId = () => `${Math.random()}`.slice(2);

export const getPrompt = (prompt: string, instruction: string) => {
  return prompt.replace(instructionString, instruction);
};
