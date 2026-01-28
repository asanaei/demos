# Demos Landing Page

## Overview
This repository contains a collection of interactive demos with a beautiful landing page to showcase them all.

## Adding a New Demo

Follow these simple steps to add a new demo:

### 1. Create a Demo Folder
Create a new folder for your demo in the root directory:
```bash
mkdir my-demo-name
```

### 2. Add Your Demo Files
Add your demo files inside the folder. At minimum, include an `index.html` file:
```
my-demo-name/
  ├── index.html
  ├── style.css (optional)
  ├── script.js (optional)
  └── ... (any other files)
```

### 3. Update the Landing Page
Edit `index.html` in the root directory and add a new demo card inside the `<div class="demos-grid">` section:

```html
<a href="my-demo-name/index.html" class="demo-card">
    <h3>My Demo Title</h3>
    <p>A brief description of what this demo does and what it demonstrates.</p>
    <div class="tags">
        <span class="tag">JavaScript</span>
        <span class="tag">CSS</span>
        <span class="tag">Animation</span>
    </div>
</a>
```

### 4. View Your Changes
Open `index.html` in a web browser to see your new demo card on the landing page.

## Demo Structure Example

```
demos/
├── index.html              # Landing page
├── example-demo/           # Example demo folder
│   └── index.html
├── my-first-demo/          # Your first demo
│   ├── index.html
│   └── style.css
└── my-second-demo/         # Your second demo
    ├── index.html
    ├── script.js
    └── assets/
        └── image.png
```

## Features

- **Responsive Design**: Works on all screen sizes
- **Easy to Expand**: Simple HTML structure for adding new demos
- **Modern UI**: Beautiful gradient design with card-based layout
- **Hover Effects**: Interactive cards with smooth transitions
- **Tag System**: Categorize demos with technology tags

## Preview

Visit the landing page by opening `index.html` in your browser, or deploy it using GitHub Pages!

## License

See the LICENSE file for details.
