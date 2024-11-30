import React from 'react';

export default function Instructions() {
  return (
    <div className="bg-red-50 p-6 rounded-lg max-w-2xl mx-auto">
      <h2 className="text-xl font-bold text-red-800 mb-4">How to Use</h2>
      <ul className="space-y-2 text-red-700">
        <li>1. Click the "Start Translation" button to begin</li>
        <li>2. Position your hand in front of the camera</li>
        <li>3. Make sign language gestures</li>
        <li>4. The translation will appear in real-time</li>
        <li>5. Click "Stop Translation" when finished</li>
      </ul>
    </div>
  );
}