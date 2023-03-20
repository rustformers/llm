import { Prompt } from "../hooks/useStore";

export const getRandomId = () => `${Math.random()}`.slice(2);

export const getPrompt = (prompt: Prompt, message: string, isFirst: boolean) => {
  if (!isFirst)
    return `${prompt.userPrefix}
  ${message}

  ${prompt.assistantPrefix}
  `;
  
  return `${prompt.instruction}
  
  ${prompt.userPrefix}
  ${message}
  
  ${prompt.assistantPrefix}
  `;
};
