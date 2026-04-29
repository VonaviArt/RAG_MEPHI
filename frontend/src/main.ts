import "./style.css";

type Chunk = {
  text: string;
  score: number;
  metadata: Record<string, unknown>;
};

type QueryResponse = {
  answer: string;
  chunks: Chunk[];
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("App root not found");
}

app.innerHTML = `
  <main class="container">
    <header class="page-header">
      <div class="brand">
        <p class="brand-kicker">Ассистент</p>
        <h1 class="page-title">RAG MEPHI</h1>
      </div>
    </header>

    <section class="section section--query" aria-labelledby="question-heading">
      <h2 id="question-heading" class="section-title">Вопрос</h2>
      <form id="queryForm" class="query-form">
        <textarea id="question" rows="5" placeholder="Привет! Чем могу помочь?" autocomplete="off"></textarea>
        <div class="actions">
          <button id="submitBtn" type="submit">Отправить</button>
        </div>
      </form>
    </section>

    <section class="section" aria-labelledby="answer-heading">
      <h2 id="answer-heading" class="section-title">Ответ</h2>
      <div id="answer" class="panel panel--muted">Задайте вопрос выше.</div>
    </section>

    <section class="section" aria-labelledby="chunks-heading">
      <h2 id="chunks-heading" class="section-title">Контекст</h2>
      <div id="chunks" class="chunks"></div>
    </section>
  </main>
`;

const form = document.querySelector<HTMLFormElement>("#queryForm");
const questionInput = document.querySelector<HTMLTextAreaElement>("#question");
const submitBtn = document.querySelector<HTMLButtonElement>("#submitBtn");
const answerBlock = document.querySelector<HTMLDivElement>("#answer");
const chunksBlock = document.querySelector<HTMLDivElement>("#chunks");

if (!form || !questionInput || !submitBtn || !answerBlock || !chunksBlock) {
  throw new Error("Required UI elements not found");
}

const chunksRoot = chunksBlock;

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderChunks(chunks: Chunk[]): void {
  if (!chunks.length) {
    chunksRoot.innerHTML = `<div class="panel panel--muted">Контекстные чанки не найдены.</div>`;
    return;
  }

  chunksRoot.innerHTML = chunks
    .map((chunk, index) => {
      const metadata = escapeHtml(JSON.stringify(chunk.metadata));
      const body = escapeHtml(chunk.text);
      return `
        <article class="panel chunk">
          <div class="chunk-header">
            <span class="chunk-label">Фрагмент ${index + 1}</span>
            <span class="chunk-score">${chunk.score.toFixed(4)}</span>
          </div>
          <p class="chunk-body">${body}</p>
          <div class="meta">${metadata}</div>
        </article>
      `;
    })
    .join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    answerBlock.textContent = "Введите вопрос.";
    answerBlock.classList.add("panel--muted");
    return;
  }

  submitBtn.disabled = true;
  answerBlock.textContent = "Обрабатываю…";
  answerBlock.classList.add("panel--muted");
  chunksRoot.innerHTML = "";

  try {
    const response = await fetch(`${API_URL}/api/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Request failed with status ${response.status}`);
    }

    const data = (await response.json()) as QueryResponse;
    answerBlock.textContent = data.answer || "Ответ пустой.";
    answerBlock.classList.remove("panel--muted");
    renderChunks(data.chunks || []);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Неизвестная ошибка";
    answerBlock.textContent = `Ошибка: ${message}`;
    answerBlock.classList.add("panel--muted");
    chunksRoot.innerHTML = "";
  } finally {
    submitBtn.disabled = false;
  }
});
