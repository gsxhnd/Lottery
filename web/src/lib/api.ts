export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    ...options,
  })
  const text = await res.text()
  let body: unknown = null

  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = { detail: text }
    }
  }

  if (!res.ok) {
    const detail = typeof body === "object" && body !== null && "detail" in body ? String((body as { detail: unknown }).detail) : res.statusText
    throw new Error(detail || `HTTP ${res.status}`)
  }

  return body as T
}
