import { createRoot } from 'react-dom/client';
import App from './App';

export default defineContentScript({
  matches: ['*://*/*'],
  cssInjectionMode: 'ui',
  async main(ctx) {
    const ui = await createShadowRootUi(ctx, {
      name: 'brightmind-chat',
      position: 'inline',
      anchor: 'body',
      onMount: (container) => {
        const root = createRoot(container);
        root.render(<App />);
        return root;
      },
      onRemove: (root) => root?.unmount(),
    });
    ui.mount();
  },
});
