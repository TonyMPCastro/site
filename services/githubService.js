const GITHUB_USERNAME = 'TonyMPCastro';
const CACHE_KEY = `gh_repos_cache_${GITHUB_USERNAME}_v1`;
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hora

/**
 * Busca repositórios públicos do GitHub com cache em localStorage e fallback.
 */
export async function fetchGitHubRepos() {
  try {
    const cachedRaw = localStorage.getItem(CACHE_KEY);
    if (cachedRaw) {
      const { timestamp, data } = JSON.parse(cachedRaw);
      if (Date.now() - timestamp < CACHE_TTL_MS && Array.isArray(data)) {
        return data;
      }
    }

    const response = await fetch(
      `https://api.github.com/users/${GITHUB_USERNAME}/repos?sort=updated&per_page=15`,
      {
        headers: {
          Accept: 'application/vnd.github.v3+json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`GitHub API HTTP status ${response.status}`);
    }

    const repos = await response.json();

    const normalizedRepos = repos
      .filter((repo) => !repo.fork && repo.name.toLowerCase() !== GITHUB_USERNAME.toLowerCase())
      .map((repo) => ({
        id: repo.id,
        name: repo.name,
        description: repo.description || 'Repositório desenvolvido por Antonio Marcos (Tony Castro).',
        htmlUrl: repo.html_url,
        homepage: repo.homepage || null,
        language: repo.language || 'Full-Stack',
        stars: repo.stargazers_count || 0,
        topics: Array.isArray(repo.topics) ? repo.topics : [],
        updatedAt: new Date(repo.pushed_at).toLocaleDateString('pt-BR', {
          month: 'short',
          year: 'numeric',
        }),
      }));

    localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({
        timestamp: Date.now(),
        data: normalizedRepos,
      })
    );

    return normalizedRepos;
  } catch (error) {
    console.warn('[Portfolio] Falha ao consultar API do GitHub. Utilizando cache expirado ou fallback estático.', error);
    const cachedFallback = localStorage.getItem(CACHE_KEY);
    if (cachedFallback) {
      return JSON.parse(cachedFallback).data;
    }
    return [];
  }
}

