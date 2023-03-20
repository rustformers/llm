import { useEffect, useRef, useState } from "react";
import { getRandomId } from "../helpers";
import { useMessage, useModel, useStore } from "../hooks/useStore";
import { useComplete } from "../hooks/useComplete";
import { invoke } from "@tauri-apps/api";
import { listen } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/api/dialog";
import { alpacaName, alpacaUrl } from "../config";

export const Main = () => {
  const model = useModel();
  if (!model) return <NoModel />;
  return (
    <div className="col-span-2 flex flex-col">
      <Title />
      <Messages />
      <Input />
    </div>
  );
};

type Progress = {
  downloaded: number;
  total_size: number;
};

const NoModel = () => {
  const addModel = useStore((state) => state.addModel);

  const download = async () => {
    const path = await save({
      defaultPath: alpacaName,
      filters: [{ name: "Model", extensions: ["bin"] }],
      title: "Where to Save?",
    });
    await invoke("download_model", { path, url: alpacaUrl });
    await addModel(path || undefined);
  };

  const [progress, setProgress] = useState<Progress>();
  useEffect(() => {
    listen<Progress>("progress", (event) => {
      setProgress(event.payload);
    });
  }, []);

  return (
    <div className="flex flex-col items-center justify-center col-span-2 space-y-2">
      {!progress ? (
        <>
          <button className="bg-zinc-300 p-2 rounded-lg" onClick={() => addModel()}>
            Select Model From Your Computer
          </button>
          <p>or</p>
          <button onClick={download} className="bg-zinc-300 p-2 rounded-lg">
            Download Alpaca
          </button>
        </>
      ) : (
        <>
          <p>Downloading...</p>
          <div className="flex items-center space-x-3">
            <p>{((progress.downloaded / progress.total_size) * 100).toFixed(2)}%</p>
            <progress className="rounded-full bg-blue-400" value={progress.downloaded} max={progress.total_size} />
          </div>
        </>
      )}
    </div>
  );
};

const Title = () => {
  const model = useModel();
  const editModel = useStore((state) => state.editModel);
  const clear = useStore((state) => state.clearMessages);
  const isActive = useStore((state) => state.isActive);
  const setIsActive = useStore((state) => state.setIsActive);
  const [loading, setLoading] = useState(false);

  const toggle = async () => {
    setLoading(true);
    if (!isActive) await invoke("start_model", { path: model?.path });
    else await invoke("stop_model");

    setLoading(false);
  };

  useEffect(() => {
    const interval = setInterval(() => invoke("is_active").then((a: any) => setIsActive(a)), 500);
    return () => clearInterval(interval);
  }, []);

  if (!model) return null;
  return (
    <div className="flex items-center justify-between p-2 border-b shadow-sm">
      <h1 className="text-lg font-medium">
        <input type="text" value={model.name} onChange={(e) => editModel(model.id, e.target.value)} />
      </h1>
      <div className="flex space-x-2 text-sm">
        <button onClick={clear}>Clear</button>
        <button className="cursor-pointer" disabled={loading} onClick={toggle}>
          {isActive ? "Stop" : "Start"}
        </button>
      </div>
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
