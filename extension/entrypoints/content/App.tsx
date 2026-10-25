import { useState } from 'react';
import './styles.css';

export default function App() {
    const [expanded, setExpanded] = useState(false);
    const [messages, setMessages] = useState<string[]>([
        "testing message 1", "testing message 2", "three testing messages"
    ]);
    const [prompt, setPrompt] = useState<string>("");

    const handleSend = () => {
        const text = prompt.trim();
        if (!text) return;
        setMessages((prev) => [...prev, text]);
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
                    {messages.map((message) => (
                        <li className="chat-message">{message}</li>
                    ))}
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