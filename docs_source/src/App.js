import React from 'react';
import './App.css';

import Header from "./components/Header";
import Section1 from "./components/Section1";
import Section2 from "./components/Section2";
import Section3 from "./components/Section3";
import Footer from "./components/Footer";

import { MathJaxContext } from 'better-react-mathjax';

const config = {
  loader: { load: ["[tex]/html"]},
  tex: {
    packages: { "[+]": ["html"] },
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"]
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"]
    ]
  }
};

function App() {
  return (
    <MathJaxContext version={3} config={config}>
      <div>
        <Header />
        <Section1 />
        <Section2 />
        <Section3 />
        <Footer />
      </div>
    </MathJaxContext>
  );
}

export default App;
