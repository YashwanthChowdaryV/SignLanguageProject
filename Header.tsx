import React from 'react';
import { SignLanguage } from 'lucide-react';

export default function Header() {
  return (
    <header className="bg-red-600 text-white py-6">
      <div className="container mx-auto px-4 flex items-center justify-center">
        <SignLanguage className="w-10 h-10 mr-3" />
        <h1 className="text-3xl font-bold">Sign Language Translator</h1>
      </div>
    </header>
  );
}