import { useState } from 'react';

export default function App() {
    const [expanded, setExpanded] = useState(false);

    return (
        <div style={{background: '#000000'}}>
            <p>Testing</p>
        </div>
    );
}