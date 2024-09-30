'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';

export default function Home() {
  const [query, setQuery] = useState<string>('');
  const [streamingData, setStreamingData] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setStreamingData('');
    setIsStreaming(true);

    const response = await fetch('/api/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: query }),
    });

    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: true });
        setStreamingData((prev) => prev + chunk);
      }
    }

    setIsStreaming(false);
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-gray-100 px-4">
      <motion.h1
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
        className="text-4xl font-bold mb-8"
      >
        Ask the AI Assistant
      </motion.h1>

      <form onSubmit={handleSubmit} className="w-full max-w-lg">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your question..."
          className="w-full p-3 rounded-lg border border-gray-300 mb-4 focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          className="w-full bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-600 transition duration-300"
        >
          Submit
        </button>
      </form>

      {isStreaming && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, loop: Infinity }}
          className="mt-8 text-lg text-blue-500"
        >
          Streaming response...
        </motion.div>
      )}

      {streamingData && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="mt-8 p-4 bg-white rounded-lg shadow-md text-lg leading-relaxed max-w-lg"
        >
          {streamingData}
        </motion.div>
      )}
    </div>
  );
}
