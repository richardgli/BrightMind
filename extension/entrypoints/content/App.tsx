import { useState } from 'react';
import './styles.css';

export default function App() {
    const [expanded, setExpanded] = useState(false);
    type Message = {role: "user" | "assistant"; text: string};
    const [messages, setMessages] = useState<Message[]>([
        {role: "assistant", text: "Hi! Ask me to quiz you or explain a topic."}
    ]);
    const [prompt, setPrompt] = useState<string>("");

    const handleSend = () => {
        const text = prompt.trim();
        if (!text) return;
        setMessages((prev) => [...prev, {role: "user", text}]);
        setPrompt("");
    }

    return (
        <div className="floating-box">
            {expanded ? (
                <div className="chat-container">
                    <header className="header">
                        <strong>BrightMind</strong>
                        <button onClick={() => setExpanded(false)}>-</button>
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
                <button className="button" onClick={() => setExpanded(true)}></button>
            )}
        </div>
    );
}