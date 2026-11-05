import { useState, useRef, useEffect } from 'react';
import './styles.css';

type Message = {role: "user" | "assistant"; text: string};

export default function App() {
    const [expanded, setExpanded] = useState(false);
    const [messages, setMessages] = useState<Message[]>([
        {role: "assistant", text: "Hi! Ask me to quiz you or explain a topic."}
    ]);
    const [prompt, setPrompt] = useState<string>("");

    const [position, setPosition] = useState({x: window.innerWidth - 60, y: window.innerHeight - 60});
    const dragging = useRef(false);
    const offset = useRef({ x: 0, y: 0 });
    const movedRef = useRef(false);

    const onDragStart = (e: React.MouseEvent) => {
        e.stopPropagation();
        dragging.current = true;
        movedRef.current = false;
        offset.current = {x: e.clientX - position.x, y: e.clientY - position.y};
    }
    
    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!dragging.current) return;
            movedRef.current = true;
            const x = Math.min(Math.max(0, e.clientX - offset.current.x), window.innerWidth - 60);
            const y = Math.min(Math.max(0, e.clientY - offset.current.y), window.innerHeight - 60);
            setPosition({x, y});
        };
        const onUp = () => { dragging.current = false; };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        return () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };
    }, [])
    
    const handleSend = () => {
        const text = prompt.trim();
        if (!text) return;
        setMessages((prev) => [...prev, {role: "user", text}]);
        setPrompt("");
    }

    return (
        <div 
            className="floating-box"
            style={{ left: position.x, top: position.y }}
        >
            {expanded ? (
                <div className="chat-container">
                    <header
                        className="header"
                        onMouseDown={onDragStart}
                        style={{ cursor: 'move' }}
                    >
                        <strong>BrightMind</strong>
                        <button
                            onMouseDown={(e) => e.stopPropagation()}
                            onClick={() => setExpanded(false)}
                        >-</button>
                    </header>
                    <ul className="chat-messages">
                        {messages.map((message, i) => (
                            <li 
                                className={`chat-message ${message.role === "assistant" ? "incoming" : ""}`}
                                key={i}
                            >
                                <div className="role-label">
                                    {message.role === "assistant" ? "BrightMind" : "You"}
                                </div>
                                <div>{message.text}</div>
                            </li>
                        ))}
                    </ul>
                    <div className="chat-box">
                        <input
                            type="text"
                            autoComplete="off"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            onKeyDown={(e) => {
                                e.stopPropagation();
                                if (e.key === "Enter") {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            onKeyUp={(e) => e.stopPropagation()}
                        />
                    </div>
                </div>
            ) : (
                <button
                    className="button"
                    onMouseDown={onDragStart}
                    onClick={() => {
                        if (!movedRef.current) setExpanded(true);
                    }}
                ></button>
            )}
        </div>
    );
}