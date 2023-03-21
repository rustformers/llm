import { useEffect, useRef, useState } from "react";
import { useListener } from "../hooks/useListener";
import { useStore } from "../hooks/useStore";

export const Input = () => {
  const ref = useRef<HTMLTextAreaElement>(null);
  const [input, setInput] = useState("");
  const addMessage = useStore((state) => state.addMessage);
  const send = useStore((state) => state.send);
  useListener();

  useEffect(() => {
    if (!ref.current) return;
    ref.current.style.height = "0px";
    const scrollHeight = ref.current.scrollHeight;
    ref.current.style.height = Math.min(scrollHeight, 200) + "px";
  }, [input]);

  const submit = () => {
    const message = addMessage(input, "user");
    send(message);
    setInput("");
  };

  return (
    <form className="p-2 relative" onSubmit={submit}>
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
