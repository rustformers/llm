import { useMessage, useStore } from "../hooks/useStore";

export const Messages = () => {
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
    <div className={`flex flex-col my-1 ${isUser ? "items-end" : "items-start"}`}>
      {message.index === 0 && (
        <div className="w-full my-2">
          <p className="text-xs text-center">New Session</p>
          <div className="w-full h-[1px] rounded-full bg-zinc-300" />
        </div>
      )}
      <p className={`rounded-lg p-2 whitespace-pre-wrap ${isUser ? "bg-blue-500 text-white" : "bg-zinc-200"}`}>
        {message.message.replace("[end of text]", "")}
      </p>
    </div>
  );
};
