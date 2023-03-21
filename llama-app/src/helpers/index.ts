import { Prompt } from "../types";

export const getRandomId = () => `${Math.random()}`.slice(2);

export const getPrompt = (prompt: Prompt, message: string, isFirst: boolean) => {
  const base = `\n\n${prompt.userPrefix}\n${message}\n\n${prompt.assistantPrefix}\n`;
  if (!isFirst) return base;

  return `${prompt.instruction}${base}`;
};
