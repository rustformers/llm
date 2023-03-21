import { Models } from "./Models";
import { Prompt } from "./Prompt";
import { Params } from "./Params";

export const Settings = () => {
  return (
    <div className="border-r h-full p-2 space-y-2 overflow-auto">
      <Models />
      <Prompt />
      <Params />
    </div>
  );
};
