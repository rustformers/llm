import { useStore } from "../../hooks/useStore";
import { Section } from "./Section";

export const Prompt = () => {
  const prompt = useStore((state) => state.prompt);
  const setPrompt = useStore((state) => state.setPrompt);
  const resetPrompt = useStore((state) => state.resetPrompt);

  return (
    <Section title="Prompt" className="flex flex-col items-center space-y- p-2">
      <div>
        <label>Instructions</label>
        <textarea
          value={prompt.instruction}
          className="min-h-[100px] w-full rounded-lg p-1"
          onChange={(e) => setPrompt({ instruction: e.target.value })}
        />
      </div>
      <div>
        <label>User prefix</label>
        <input value={prompt.userPrefix} className="w-full rounded-lg p-1" onChange={(e) => setPrompt({ userPrefix: e.target.value })} />
      </div>
      <div>
        <label>Assistant prefix</label>
        <input value={prompt.assistantPrefix} className="w-full rounded-lg p-1" onChange={(e) => setPrompt({ assistantPrefix: e.target.value })} />
      </div>

      <button className="hover:bg-zinc-300 p-2 rounded-lg" onClick={resetPrompt}>
        Reset
      </button>
    </Section>
  );
};
