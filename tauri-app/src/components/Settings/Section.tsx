import { useState } from "react";
import { IoIosArrowDown } from "react-icons/io";

export const Section = ({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) => {
    const [open, setOpen] = useState(true);
    return (
      <div className="flex flex-col bg-zinc-200 rounded-lg overflow-hidden">
        <div className="flex justify-between items-center p-2 cursor-pointer bg-zinc-300" onClick={() => setOpen(!open)}>
          <h3 className="text-lg font-semibold text-primary-content">{title}</h3>
          <IoIosArrowDown className={`duration-150 ${open ? "-rotate-180" : ""}`} />
        </div>
        {open && <div className={className}>{children}</div>}
      </div>
    );
  };
  