import React, { useState } from 'react';
import { Camera } from 'lucide-react';

export default function TranslatorBox() {
  const [translation, setTranslation] = useState<string>('');
  const [isRecording, setIsRecording] = useState(false);

  const startTranslation = () => {
    setIsRecording(true);
    // In a real implementation, this would connect to your Python backend
    // and start the webcam capture/translation process
  };

  const stopTranslation = () => {
    setIsRecording(false);
    // This would stop the translation process
  };

  return (
    <div className="bg-yellow-100 p-8 rounded-lg shadow-lg max-w-2xl mx-auto">
      <div className="bg-white p-6 rounded-lg mb-6 min-h-[300px] flex items-center justify-center">
        {!isRecording ? (
          <div className="text-center">
            <Camera className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <p className="text-gray-500">Camera feed will appear here</p>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-green-600">Recording...</p>
          </div>
        )}
      </div>
      
      <div className="text-center">
        <button
          onClick={isRecording ? stopTranslation : startTranslation}
          className={`px-6 py-3 rounded-full font-semibold text-white transition-colors ${
            isRecording 
              ? 'bg-red-500 hover:bg-red-600' 
              : 'bg-yellow-500 hover:bg-yellow-600'
          }`}
        >
          {isRecording ? 'Stop Translation' : 'Start Translation'}
        </button>
      </div>

      <div className="mt-6 p-4 bg-white rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Translation:</h3>
        <p className="text-gray-700 min-h-[50px]">
          {translation || 'Translation will appear here...'}
        </p>
      </div>
    </div>
  );
}