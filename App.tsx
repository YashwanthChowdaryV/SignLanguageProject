import React from 'react';
import Header from './components/Header';
import TranslatorBox from './components/TranslatorBox';
import Instructions from './components/Instructions';

function App() {
  return (
    <div className="min-h-screen bg-red-100">
      <Header />
      <main className="container mx-auto px-4 py-8 space-y-8">
        <TranslatorBox />
        <Instructions />
      </main>
    </div>
  );
}

export default App;