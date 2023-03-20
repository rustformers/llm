import { useEffect, useRef, useState } from "react";
import { useComplete } from "../hooks/useComplete";
import { useStore } from "../hooks/useStore";

export const Input = () => {
  const ref = useRef<HTMLTextAreaElement>(null);
  const [input, setInput] = useState("");
  const addMessage = useStore((state) => state.addMessage);
  const isGenerating = useStore((state) => state.isGenerating);
  const send = useComplete();

  useEffect(() => {
    if (!ref.current) return;
    ref.current.style.height = "0px";
    const scrollHeight = ref.current.scrollHeight;
    ref.current.style.height = Math.min(scrollHeight, 200) + "px";
  }, [input]);

  const submit = () => {
    if (isGenerating) return;
    addMessage(input, "user");
    send(input);
    setInput("");
  };

  return (
    <form className="p-2 relative" onSubmit={submit}>
      {isGenerating && (
        <div className="absolute top-[-10px] text-center text-xs w-full">
          <p>Generating...</p>
        </div>
      )}
      <textarea
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        ref={ref}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="bg-zinc-300 w-full p-2 rounded-lg resize-none"
      />
    </form>
  );
};
