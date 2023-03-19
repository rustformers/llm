import { useEffect } from "react";
import { Main } from "./components/Main";
import { Menu } from "./components/Menu";

export default function App() {
  async function greet() {
    // set({ answer: "" });
    // invoke("complete", { input: { ...input, path, prompt } });
  }
  useEffect(() => {
    // listen("message", (event) => {
    //   set({ answer: (event.payload as any)?.message });
    // });
  }, []);

  return (
    <div className="grid grid-cols-3 h-screen overflow-hidden">
      <Menu />
      <Main />
    </div>
  );
}
