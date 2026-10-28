/* ==========================================================================
   ANTONIO MARCOS PATRICIO CASTRO - MODERN PORTFOLIO INTERACTIVITY
   ========================================================================== */

import { FEATURED_CASES } from '../data/featuredCases.js';
import { fetchGitHubRepos } from '../services/githubService.js';

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initMobileMenu();
  initFeaturedCases();
  initGitHubRepos();
  initScrollReveal();
  initSmoothScroll();
  initCounterStats();
});

/* Copy Email to Clipboard with Visual Feedback (No broken mailto:) */
export async function copyEmailToClipboard(email, buttonElement) {
  const originalHTML = buttonElement.innerHTML;
  try {
    await navigator.clipboard.writeText(email);
    buttonElement.innerHTML = '✅ E-mail copiado!';
    buttonElement.disabled = true;
    setTimeout(() => {
      buttonElement.innerHTML = originalHTML;
      buttonElement.disabled = false;
    }, 2500);
  } catch (err) {
    // Fallback caso a API de clipboard seja bloqueada
    window.location.href = `mailto:${email}`;
  }
}

window.copyEmailToClipboard = copyEmailToClipboard;

/* Render Hybrid Featured Cases & Filter Tabs */
function initFeaturedCases() {
  const grid = document.getElementById('featured-cases-grid');
  const filterButtons = document.querySelectorAll('.filter-btn');
  if (!grid) return;

  const renderCases = (category = 'all') => {
    const filtered =
      category === 'all'
        ? FEATURED_CASES
        : FEATURED_CASES.filter((item) => item.category === category);

    grid.innerHTML = filtered
      .map((item) => {
        const privacyTag = item.isPrivateRepo
          ? `<span class="case-privacy-tag private"><i class="fa-solid fa-lock" aria-hidden="true"></i> Código Privado (NDA/Comercial)</span>`
          : `<span class="case-privacy-tag public"><i class="fa-solid fa-lock-open" aria-hidden="true"></i> Open Source</span>`;

        const metricsHTML = item.metrics
          .map(
            (metric) =>
              `<span class="case-metric-pill"><i class="fa-solid fa-check-circle" aria-hidden="true"></i> ${metric}</span>`
          )
          .join('');

        const stackHTML = item.stack
          .map((tech) => `<span class="skill-tag">${tech}</span>`)
          .join('');

        const actionLink = item.repoUrl
          ? `<a href="${item.repoUrl}" target="_blank" rel="noopener noreferrer" class="case-link" aria-label="Ver repositório de ${item.title} no GitHub">
               <i class="fa-brands fa-github" aria-hidden="true"></i> Ver no GitHub <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i>
             </a>`
          : `<span class="case-link" style="color: var(--text-dim);">
               <i class="fa-solid fa-shield-halved" aria-hidden="true"></i> Arquitetura Proprietária (Código Privado)
             </span>`;

        return `
          <article class="glass-card case-card reveal active">
            <div>
              <div class="case-header">
                <span class="case-badge">${item.badge}</span>
                ${privacyTag}
              </div>
              <h3 class="case-title" style="margin-top: 1rem; margin-bottom: 0.85rem;">${item.title}</h3>
              <p class="case-block" style="margin-bottom: 0.65rem;">
                <strong>Desafio:</strong> ${item.problem}
              </p>
              <p class="case-block" style="margin-bottom: 1.1rem;">
                <strong>Solução:</strong> ${item.solution}
              </p>
              <div class="case-metrics" style="margin-bottom: 1.1rem;">
                ${metricsHTML}
              </div>
            </div>
            <div>
              <div class="skill-tags" style="margin-bottom: 1rem;">
                ${stackHTML}
              </div>
              <div class="case-footer">
                ${actionLink}
              </div>
            </div>
          </article>
        `;
      })
      .join('');
  };

  renderCases('all');

  filterButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      filterButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const category = btn.getAttribute('data-filter') || 'all';
      renderCases(category);
    });
  });
}

/* Fetch & Render Public GitHub Repositories with localStorage Cache */
async function initGitHubRepos() {
  const reposGrid = document.getElementById('github-repos-grid');
  if (!reposGrid) return;

  const repos = await fetchGitHubRepos();

  if (!repos || repos.length === 0) {
    reposGrid.innerHTML = `
      <div class="glass-card" style="grid-column: 1 / -1; text-align: center;">
        <p style="color: var(--text-muted);">
          Explore todos os repositórios públicos diretamente no 
          <a href="https://github.com/TonyMPCastro" target="_blank" rel="noopener noreferrer">GitHub (@TonyMPCastro)</a>.
        </p>
      </div>
    `;
    return;
  }

  reposGrid.innerHTML = repos
    .slice(0, 6)
    .map(
      (repo) => `
      <article class="glass-card repo-card reveal active">
        <div>
          <div class="repo-header">
            <a href="${repo.htmlUrl}" target="_blank" rel="noopener noreferrer" class="repo-name" aria-label="Abrir repositório ${repo.name} no GitHub">
              <i class="fa-solid fa-book-bookmark" style="color: var(--primary); margin-right: 0.4rem;" aria-hidden="true"></i>${repo.name}
            </a>
            ${
              repo.stars > 0
                ? `<span class="repo-stars" aria-label="${repo.stars} estrelas"><i class="fa-solid fa-star" aria-hidden="true"></i> ${repo.stars}</span>`
                : ''
            }
          </div>
          <p class="repo-desc" style="margin-top: 0.65rem;">${repo.description}</p>
        </div>
        <div class="repo-footer">
          <span class="repo-lang"><i class="fa-solid fa-code" aria-hidden="true"></i> ${repo.language}</span>
          <span class="repo-date">Atualizado em ${repo.updatedAt}</span>
        </div>
      </article>
    `
    )
    .join('');
}

/* Sticky Navbar on Scroll */
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;

  const handleScroll = () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
    highlightActiveNavLink();
  };

  window.addEventListener('scroll', handleScroll);
  handleScroll();
}

/* Mobile Hamburger Menu */
function initMobileMenu() {
  const hamburger = document.querySelector('.hamburger');
  const navLinks = document.querySelector('.nav-links');
  const links = document.querySelectorAll('.nav-link, .nav-btn');

  if (!hamburger || !navLinks) return;

  hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    const icon = hamburger.querySelector('i');
    if (icon) {
      if (navLinks.classList.contains('open')) {
        icon.classList.remove('fa-bars');
        icon.classList.add('fa-xmark');
      } else {
        icon.classList.remove('fa-xmark');
        icon.classList.add('fa-bars');
      }
    }
  });

  links.forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
      const icon = hamburger.querySelector('i');
      if (icon) {
        icon.classList.remove('fa-xmark');
        icon.classList.add('fa-bars');
      }
    });
  });
}

/* Scroll Reveal Animations using IntersectionObserver */
function initScrollReveal() {
  const reveals = document.querySelectorAll('.reveal');
  if (!reveals.length) return;

  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
          obs.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.15,
      rootMargin: '0px 0px -50px 0px',
    }
  );

  reveals.forEach((el) => observer.observe(el));
}

/* Active Nav Link Highlight based on scroll position */
function highlightActiveNavLink() {
  const sections = document.querySelectorAll('section[id]');
  const scrollY = window.pageYOffset;

  sections.forEach((current) => {
    const sectionHeight = current.offsetHeight;
    const sectionTop = current.offsetTop - 120;
    const sectionId = current.getAttribute('id');
    const navLink = document.querySelector(`.nav-link[href*="${sectionId}"]`);

    if (navLink) {
      if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
        navLink.classList.add('active');
      } else {
        navLink.classList.remove('active');
      }
    }
  });
}

/* Smooth Scrolling for internal anchors */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      const targetElement = document.querySelector(targetId);

      if (targetElement) {
        e.preventDefault();
        const headerOffset = 80;
        const elementPosition = targetElement.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth',
        });
      }
    });
  });
}

/* Animated Number Counter for stats */
function initCounterStats() {
  const counters = document.querySelectorAll('.counter-num');
  if (!counters.length) return;

  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const counter = entry.target;
          const target = +counter.getAttribute('data-target');
          const duration = 2000;
          const stepTime = 20;
          const steps = duration / stepTime;
          const increment = target / steps;
          let current = 0;

          const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
              counter.innerText = target;
              clearInterval(timer);
            } else {
              counter.innerText = Math.ceil(current);
            }
          }, stepTime);

          obs.unobserve(counter);
        }
      });
    },
    { threshold: 0.5 }
  );

  counters.forEach((c) => observer.observe(c));
}
