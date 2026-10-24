import { useState } from 'react';
import './styles.css';

export default function App() {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className="floating-box">
            {expanded ? (
                <div className="chat-container">
                    <header className="header">
                        <strong>BrightMind</strong>
                        <button onClick={() => setExpanded(false)}>-</button>
                    </header>
                </div>
            ) : (
                <button className="button" onClick={() => setExpanded(true)}></button>
            )}
        </div>
    );
}