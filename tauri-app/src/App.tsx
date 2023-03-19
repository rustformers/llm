import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";
import { listen } from "@tauri-apps/api/event";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { open } from "@tauri-apps/api/dialog";

type Store = {
  path?: string;
  prompt: string;
  answer: string;
  set: (partial: Store | Partial<Store> | ((state: Store) => Store | Partial<Store>), replace?: boolean | undefined) => void;
};

const input = {
  n_batch: 8,
  n_threads: 4,
  top_k: 40,
  top_p: 0.95,
  repeat_penalty: 1.3,
  temp: 0.8,
  num_predict: 512,
};

const useStore = create(persist<Store>((set) => ({ set, prompt: "", answer: "" }), { name: "llama-rs" }));

export default function App() {
  const path = useStore((state) => state.path);
  const prompt = useStore((state) => state.prompt);
  const answer = useStore((state) => state.answer);
  const set = useStore((state) => state.set);

  async function greet() {
    set({ answer: "" });
    invoke("complete", { input: { ...input, path, prompt } });
  }
  useEffect(() => {
    listen("message", (event) => {
      set({ answer: (event.payload as any)?.message });
    });
  }, []);

  return (
    <div className="container">
      <SelectModel />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          greet();
        }}
      >
        <input id="greet-input" onChange={(e) => set({ prompt: e.currentTarget.value })} placeholder="Enter a name..." />
        <button type="submit">Greet</button>
      </form>
      <p>{answer}</p>
    </div>
  );
}

const SelectModel = () => {
  const set = useStore((state) => state.set);
  const path = useStore((state) => state.path);
  const select = async () => {
    const path = await open({
      directory: false,
      multiple: false,
      title: "Select Model",
      filters: [{ name: "Model", extensions: ["bin"] }],
    });
    if (!path || Array.isArray(path)) return;
    set({ path: path });
  };
  return <button onClick={select}>{path ? `Model: ${path.split("/").pop()}` : "Select Model"}</button>;
};
