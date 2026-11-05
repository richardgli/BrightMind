import { useState, useRef, useEffect } from 'react';
import './styles.css';

type Message = {role: "user" | "assistant"; text: string};

export default function App() {
    const [expanded, setExpanded] = useState(false);
    const [closing, setClosing] = useState(false);
    const [messages, setMessages] = useState<Message[]>([
        {role: "assistant", text: "Hi! Ask me to quiz you or explain a topic."}
    ]);
    const [prompt, setPrompt] = useState<string>("");

    const [position, setPosition] = useState({x: window.innerWidth - 60, y: window.innerHeight - 60});
    const dragging = useRef(false);
    const offset = useRef({ x: 0, y: 0 });
    const movedRef = useRef(false);
    const isRight = position.x + 30 > window.innerWidth / 2;
    const isBottom = position.y + 30 > window.innerHeight / 2;
    const transformOrigin = `${isBottom ? 'bottom' : 'top'} ${isRight ? 'right' : 'left'}`;

    const boxStyle: React.CSSProperties = {
        left:   isRight  ? undefined : position.x,
        right:  isRight  ? window.innerWidth - position.x - 56 : undefined,
        top:    isBottom ? undefined : position.y,
        bottom: isBottom ? window.innerHeight - position.y - 56 : undefined,
    };

    const closePanel = () => {
        setClosing(true);
        setTimeout(() => {
            setExpanded(false);
            setClosing(false);
        }, 150);
    };

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
            style={boxStyle}
        >
            {expanded ? (
                <div
                    className={`chat-container ${closing ? 'closing' : ''}`}
                    style={{ transformOrigin }}
                >
                    <header
                        className="header"
                        onMouseDown={onDragStart}
                        style={{ cursor: 'move' }}
                    >
                        <strong>BrightMind</strong>
                        <button
                            onMouseDown={(e) => e.stopPropagation()}
                            onClick={() => closePanel()}
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