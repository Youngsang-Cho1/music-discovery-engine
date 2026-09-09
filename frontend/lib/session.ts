const SESSION_KEY = "mde.session_id";

/**
 * Stable per-browser session id.
 *
 * The bandit assigns an arm per request, but attribution needs a subject to
 * group events by — otherwise the same listener's skips and completes look
 * like independent users and the posteriors update on correlated evidence.
 */
export function getSessionId(): string {
  if (typeof window === "undefined") return "server";

  let id = window.localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    window.localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}
