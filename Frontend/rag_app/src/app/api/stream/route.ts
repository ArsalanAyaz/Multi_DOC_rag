import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  const { question } = await req.json();

  const fastApiResponse = await fetch('http://127.0.0.1:8000/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  });

  if (!fastApiResponse.body) {
    return NextResponse.json({ error: 'No response from server' }, { status: 500 });
  }

  const reader = fastApiResponse.body.getReader();
  const decoder = new TextDecoder();
  let done = false;
  const bodyStream = new ReadableStream({
    async pull(controller) {
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: true });
        controller.enqueue(chunk);
        if (done) {
          controller.close();
        }
      }
    }
  });

  return new NextResponse(bodyStream);
}
