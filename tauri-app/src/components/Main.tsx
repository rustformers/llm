import { useEffect, useRef, useState } from "react";
import { getRandomId } from "../helpers";
import { useMessage, useModel, useStore } from "../hooks/useStore";
import { IoIosRefresh as Clear } from "react-icons/io";
import { useComplete } from "../hooks/useComplete";

export const Main = () => {
  return (
    <div className="col-span-2 flex flex-col">
      <Title />
      <Messages />
      <Input />
    </div>
  );
};

const Title = () => {
  const model = useModel();
  const editModel = useStore((state) => state.editModel);
  const addModel = useStore((state) => state.addModel);
  const clear = useStore((state) => state.clearMessages);

  return (
    <div className="flex items-center justify-between p-2 border-b shadow-sm">
      <div />
      <h1 className="text-xl font-medium">
        {model ? (
          <input type="text" className="text-center" value={model.name} onChange={(e) => editModel({ ...model, name: e.target.value })} />
        ) : (
          <span className="cursor-pointer text-red-400" onClick={addModel}>
            No Model Selected
          </span>
        )}
      </h1>
      <Clear className="text-xl cursor-pointer" onClick={clear} />
    </div>
  );
};

const Messages = () => {
  const allMessages = useStore((state) => state.allMessages);
  return (
    <div className="relative mb-auto h-full">
      <div className="top-0 left-0 w-full absolute flex flex-col-reverse px-2 h-full mb-auto overflow-y-auto overflow-x-hidden">
        {[...allMessages].reverse().map((id) => (
          <Message key={id} id={id} />
        ))}
      </div>
    </div>
  );
};

const Message = ({ id }: { id: string }) => {
  const message = useMessage(id);
  if (!message) return null;
  const isUser = message.type === "user";
  return (
    <div className={`flex my-1 ${isUser ? "justify-end" : ""}`}>
      <p className={`rounded-lg p-2 whitespace-pre-wrap ${isUser ? "bg-blue-500 text-white" : "bg-zinc-200"}`}>
        {message.message.replace("[end of text]", "")}
      </p>
    </div>
  );
};

const Input = () => {
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
    const id = getRandomId();
    addMessage({ id, type: "user", message: input });
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
