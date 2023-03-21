import { useModel } from "../hooks/useStore";
import { Title } from "./Title";
import { NoModel } from "./NoModel";
import { Messages } from "./Messages";
import { Input } from "./Input";

export const Main = () => {
  const model = useModel();
  if (!model) return <NoModel />;
  return (
    <div className="col-span-2 flex flex-col h-screen">
      <Title />
      <Messages />
      <Input />
    </div>
  );
};
