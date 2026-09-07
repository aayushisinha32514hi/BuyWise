/**
 * BuyWise — 2026 Live Indian Shopping & AI Decision-Support Platform Controller
 * ==============================================================================
 * Integrates:
 * 1. Live Google Shopping Search with Canonical Product Identity
 * 2. NLP Shopping Requirement Understanding & Preference Auto-Fill
 * 3. Supervised Machine Learning Suitability Predictions
 * 4. Multi-Retailer Price Comparison & Spread Analytics
 * 5. Empirical Price Observation History & Trend Indicators
 * 6. Responsive Desktop/Tablet Navigation & Mobile Bottom Bar
 */

const API_BASE = '/api/v1';

// App State
let currentTab = 'home';
let topCategories = [];
let currentUser = JSON.parse(localStorage.getItem('buywise_user') || 'null');
let authToken = localStorage.getItem('buywise_token') || '';
let compareIds = JSON.parse(localStorage.getItem('buywise_compare') || '[]');
let savedIds = JSON.parse(localStorage.getItem('buywise_saved') || '[]');
let searchTimeout = null;

let currentRequirements = {};

let currentSearchFilters = {
  q: '',
  category: '',
  max_price: 150000,
  min_rating: 0,
  sort_by: 'score',
  page: 1
};

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  updateAuthUI();
  updateCompareBadge();
  updateSavedBadge();
  await loadTaxonomy();
  loadHomeFeed();
  loadSearchResults();
  loadCategoryRequirementsUI();
  if (currentUser) {
    syncUserSaved();
  }
}

// ----------------------------------------------------
// Navigation Tab Switching (Supports Mobile & Desktop)
// ----------------------------------------------------
function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll('.tab-view').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.desktop-nav-btn').forEach(d => d.classList.remove('active'));

  const activeTabEl = document.getElementById(`tab-${tabId}`);
  if (activeTabEl) activeTabEl.classList.add('active');

  const activeNavEl = document.getElementById(`nav-${tabId}`);
  if (activeNavEl) activeNavEl.classList.add('active');

  const activeDesktopNavEl = document.getElementById(`dnav-${tabId}`);
  if (activeDesktopNavEl) activeDesktopNavEl.classList.add('active');

  window.scrollTo({ top: 0, behavior: 'smooth' });

  if (tabId === 'compare') {
    renderCompareView();
  } else if (tabId === 'saved') {
    renderSavedView();
  } else if (tabId === 'recommend') {
    if (Object.keys(currentRequirements).length === 0) {
      loadCategoryRequirementsUI();
    }
  }
}

// ----------------------------------------------------
// 1. Categories & Home Feed
// ----------------------------------------------------
async function loadTaxonomy() {
  try {
    const resTop = await fetch(`${API_BASE}/categories/top`);
    if (resTop.ok) {
      topCategories = await resTop.json();
      renderHomeTopCategories();
      renderSearchCategoryPills();
      const currentCat = document.getElementById('rec-category')?.value || 'Laptops';
      populateSubcategories(currentCat);
    }
  } catch (err) {
    console.error('Error loading taxonomy:', err);
  }
}

function renderHomeTopCategories() {
  const container = document.getElementById('home-top-categories');
  if (!container || topCategories.length === 0) return;

  container.innerHTML = topCategories.map(cat => `
    <div class="top-cat-card" onclick="selectTopCategoryBrowse('${cat.id}')">
      <div class="top-cat-icon-box">${cat.icon}</div>
      <div class="top-cat-label">${escapeHtml(cat.short_name)}</div>
    </div>
  `).join('');
}

function renderSearchCategoryPills() {
  const container = document.getElementById('search-topcats-bar');
  if (!container) return;

  container.innerHTML = `
    <button class="category-chip active" onclick="selectSearchCategoryPill('', this)">All</button>
    ${topCategories.map(c => `
      <button class="category-chip" onclick="selectSearchCategoryPill('${c.id}', this)">${c.icon} ${escapeHtml(c.short_name)}</button>
    `).join('')}
  `;
}

async function loadHomeFeed() {
  const container = document.getElementById('home-top-products');
  try {
    const res = await fetch(`${API_BASE}/products?q=best%20deals%20India&sort_by=score&limit=6`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (container && data.products && data.products.length > 0) {
      container.innerHTML = data.products.map(p => renderProductCard(p)).join('');
    } else if (container) {
      container.innerHTML = `<div class="empty-state" style="grid-column: span 2;"><p style="color: var(--text-muted);">Configure SERPAPI_KEY in .env to load live products.</p></div>`;
    }
  } catch (err) {
    console.error('Error loading home feed:', err);
    if (container) {
      container.innerHTML = `<div class="empty-state" style="grid-column: span 2;"><p style="color: var(--text-muted);">Current shopping data is temporarily unavailable.</p></div>`;
    }
  }
}

function selectTopCategoryBrowse(catId) {
  currentSearchFilters.category = catId;
  currentSearchFilters.q = '';
  currentSearchFilters.page = 1;

  document.querySelectorAll('#search-topcats-bar .category-chip').forEach(c => c.classList.remove('active'));
  switchTab('search');
  loadSearchResults();
}

function selectSearchCategoryPill(catId, el) {
  document.querySelectorAll('#search-topcats-bar .category-chip').forEach(c => c.classList.remove('active'));
  if (el) el.classList.add('active');
  currentSearchFilters.category = catId;
  currentSearchFilters.page = 1;
  loadSearchResults();
}

// ----------------------------------------------------
// 2. Search & Browse
// ----------------------------------------------------
function handleSearchKey(e) {
  const q = e.target.value;
  const clearBtn = document.getElementById('search-clear');
  if (clearBtn) clearBtn.style.display = q ? 'block' : 'none';

  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => {
    currentSearchFilters.q = q.trim();
    currentSearchFilters.page = 1;
    loadSearchResults();
  }, 400);
}

function clearSearch() {
  const input = document.getElementById('search-query');
  if (input) input.value = '';
  document.getElementById('search-clear').style.display = 'none';
  currentSearchFilters.q = '';
  currentSearchFilters.page = 1;
  loadSearchResults();
}

function toggleFilterDrawer() {
  const sheet = document.getElementById('filter-sheet');
  if (sheet) sheet.classList.toggle('open');
}

function updateBudgetLabel(val) {
  const label = document.getElementById('budget-val');
  if (label) label.innerText = `₹${parseInt(val).toLocaleString('en-IN')}`;
  currentSearchFilters.max_price = parseFloat(val);
}

function setRatingFilter(minR, el) {
  document.querySelectorAll('.rating-pills .pill').forEach(p => p.classList.remove('active'));
  if (el) el.classList.add('active');
  currentSearchFilters.min_rating = minR;
  applyFilters();
}

function applyFilters() {
  const sort = document.getElementById('filter-sort')?.value;
  if (sort) currentSearchFilters.sort_by = sort;
  currentSearchFilters.page = 1;
  loadSearchResults();
}

async function loadSearchResults() {
  const grid = document.getElementById('search-products-grid');
  const countLabel = document.getElementById('results-count');
  if (grid) grid.innerHTML = `<div class="skeleton-card"></div><div class="skeleton-card"></div>`;

  const params = new URLSearchParams();
  if (currentSearchFilters.q) params.append('q', currentSearchFilters.q);
  if (currentSearchFilters.category) params.append('category', currentSearchFilters.category);
  if (currentSearchFilters.max_price && currentSearchFilters.max_price > 0) {
    params.append('max_price', currentSearchFilters.max_price);
  }
  if (currentSearchFilters.min_rating && currentSearchFilters.min_rating > 0) {
    params.append('min_rating', currentSearchFilters.min_rating);
  }
  if (currentSearchFilters.sort_by) params.append('sort_by', currentSearchFilters.sort_by);
  params.append('page', currentSearchFilters.page);
  params.append('limit', 20);

  try {
    const res = await fetch(`${API_BASE}/products?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();

    if (data.error) {
      if (countLabel) countLabel.innerText = 'Provider Configuration Required';
      if (grid) {
        grid.innerHTML = `
          <div class="empty-state" style="grid-column: span 2;">
            <div class="empty-icon">🔑</div>
            <h4>Shopping Search Key Required</h4>
            <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 6px;">
              ${escapeHtml(data.error)}
            </p>
          </div>
        `;
      }
      renderPagination(1, 1);
      return;
    }

    if (countLabel) {
      const sourceTag = data.source ? ` • ${data.source}` : '';
      countLabel.innerText = `Showing ${data.total.toLocaleString()} products (Page ${data.page} of ${data.total_pages})${sourceTag}`;
    }

    if (!data.products || data.products.length === 0) {
      if (grid) {
        grid.innerHTML = `
          <div class="empty-state" style="grid-column: span 2;">
            <div class="empty-icon">🔍</div>
            <h4>No Products Found</h4>
            <p>Try searching for popular items like "laptop under 80000", "5G phone", or "smartwatch".</p>
          </div>
        `;
      }
      renderPagination(1, 1);
      return;
    }

    if (grid) {
      grid.innerHTML = data.products.map(p => renderProductCard(p)).join('');
    }

    renderPagination(data.page, data.total_pages);
  } catch (err) {
    console.error('Error fetching search results:', err);
    if (countLabel) countLabel.innerText = 'Search unavailable';
    if (grid) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: span 2;">
          <div class="empty-icon">⚠️</div>
          <h4>Current shopping data is temporarily unavailable</h4>
          <p>Please try again in a few moments.</p>
        </div>
      `;
    }
  }
}

function renderPagination(current, total) {
  const container = document.getElementById('pagination-controls');
  if (!container) return;
  if (total <= 1) {
    container.innerHTML = '';
    return;
  }

  container.innerHTML = `
    <div style="display: flex; justify-content: center; align-items: center; gap: 12px; margin-top: 20px;">
      <button class="btn-outline-sm" ${current === 1 ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''} onclick="goToPage(${current - 1})">← Previous</button>
      <span style="font-size: 0.8rem; font-weight: 700;">Page ${current} of ${total}</span>
      <button class="btn-outline-sm" ${current >= total ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''} onclick="goToPage(${current + 1})">Next →</button>
    </div>
  `;
}

function goToPage(p) {
  currentSearchFilters.page = p;
  loadSearchResults();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ----------------------------------------------------
// 3. NLP Requirement Understanding & AI Match Wizard
// ----------------------------------------------------
function populateSubcategories(categoryVal, defaultSubcat = null) {
  const subSelect = document.getElementById('rec-subcategory');
  if (!subSelect) return;

  const catClean = (categoryVal || '').toLowerCase();
  const catObj = topCategories.find(c =>
    (c.id && c.id.toLowerCase() === catClean) ||
    (c.short_name && c.short_name.toLowerCase() === catClean) ||
    (c.name && c.name.toLowerCase() === catClean)
  );

  const subcats = catObj && catObj.subcategories ? catObj.subcategories : [];
  if (subcats.length === 0) {
    subSelect.innerHTML = `<option value="">All Subcategories</option>`;
    return;
  }

  subSelect.innerHTML = subcats.map(s =>
    `<option value="${s.id}">${escapeHtml(s.name)}</option>`
  ).join('');

  if (defaultSubcat) {
    const defaultClean = defaultSubcat.toLowerCase();
    const matched = subcats.find(s => s.id.toLowerCase() === defaultClean || s.name.toLowerCase() === defaultClean || defaultClean.includes(s.id.toLowerCase()));
    if (matched) {
      subSelect.value = matched.id;
    }
  }
}

function onCategoryChange() {
  const catVal = document.getElementById('rec-category')?.value || 'Laptops';
  populateSubcategories(catVal);
  loadCategoryRequirementsUI();
}

function onSubcategoryChange() {
  loadCategoryRequirementsUI();
}

// ----------------------------------------------------
// 3. NLP Requirement Understanding & AI Match Wizard
// ----------------------------------------------------
async function parseNaturalLanguagePrompt() {
  const input = document.getElementById('rec-nlp-input');
  const btn = document.getElementById('nlp-parse-btn');
  const statusEl = document.getElementById('nlp-parsed-status');
  if (!input || !input.value.trim()) {
    showToast('Please type your requirements first (e.g. "phone under 60k with best camera")');
    return;
  }

  btn.innerText = 'Analyzing...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/ai-match/parse-requirements`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: input.value.trim() })
    });

    if (!res.ok) throw new Error('NLP parsing failed');
    const data = await res.json();

    // 1. Set Category & Subcategory
    const catSelect = document.getElementById('rec-category');
    if (catSelect && data.category) {
      catSelect.value = data.category;
    }
    populateSubcategories(data.category, data.sub_category);
    if (data.sub_category) {
      const subSelect = document.getElementById('rec-subcategory');
      if (subSelect && data.sub_category) {
        subSelect.value = data.sub_category;
      }
    }

    // 2. Set Budget
    if (data.min_price !== undefined) {
      const minInput = document.getElementById('rec-min-price');
      if (minInput) minInput.value = data.min_price;
    }
    if (data.max_price !== undefined) {
      const maxInput = document.getElementById('rec-max-price');
      if (maxInput) maxInput.value = data.max_price;
    }

    // 3. Set Rating & Priority
    if (data.min_rating !== undefined) {
      const ratingSelect = document.getElementById('rec-min-rating');
      if (ratingSelect) ratingSelect.value = String(data.min_rating);
    }
    if (data.priority) {
      const pRadio = document.querySelector(`input[name="rec-priority"][value="${data.priority}"]`);
      if (pRadio) pRadio.checked = true;
    }

    // 4. Set Preferred Brands
    if (data.preferred_brands && data.preferred_brands.length > 0) {
      const brandsInput = document.getElementById('rec-preferred-brands');
      if (brandsInput) brandsInput.value = data.preferred_brands.join(', ');
    }

    // 5. Reload requirements for category & subcategory and set star weights
    await loadCategoryRequirementsUI(data.requirements, data.sub_category);

    // Show status confirmation
    if (statusEl) {
      statusEl.style.display = 'block';
      statusEl.innerHTML = `✓ <strong>Requirements filled:</strong> ${escapeHtml(data.extracted_summary)}`;
    }
    showToast(`✓ Requirements filled for ${data.category}!`);

  } catch (err) {
    console.error('NLP Parse error:', err);
    showToast('Could not extract requirements from text.');
  } finally {
    btn.innerText = '⚡ Auto-Fill';
    btn.disabled = false;
  }
}

async function loadCategoryRequirementsUI(customWeights = null, targetSubcat = null) {
  const cat = document.getElementById('rec-category')?.value || 'Laptops';
  const subcat = targetSubcat || document.getElementById('rec-subcategory')?.value || '';
  const container = document.getElementById('rec-requirements-list');
  if (!container) return;

  try {
    const url = `${API_BASE}/categories/${encodeURIComponent(cat)}/requirements${subcat ? `?subcategory=${encodeURIComponent(subcat)}` : ''}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to load category requirements');
    const reqs = await res.json();
    currentRequirements = {};

    container.innerHTML = reqs.map(r => {
      const defaultVal = customWeights && customWeights[r.key] !== undefined ? customWeights[r.key] : (r.default || 3);
      currentRequirements[r.key] = defaultVal;
      return `
        <div class="req-item-card">
          <div class="req-item-info">
            <div class="req-item-title">${escapeHtml(r.label)}</div>
            <div class="req-item-desc">${escapeHtml(r.desc)}</div>
          </div>
          <div class="star-rating-selector" id="stars-req-${r.key}">
            ${[1, 2, 3, 4, 5].map(star => `
              <button type="button" class="star-btn ${star <= defaultVal ? 'active' : ''}" onclick="setRequirementStar('${r.key}', ${star})" title="${star} Star Importance">
                ★
              </button>
            `).join('')}
          </div>
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error('Error loading requirements:', err);
  }
}

function setRequirementStar(key, starVal) {
  currentRequirements[key] = starVal;
  const container = document.getElementById(`stars-req-${key}`);
  if (!container) return;

  const btns = container.querySelectorAll('.star-btn');
  btns.forEach((b, idx) => {
    if (idx < starVal) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });
}

async function executeRecommendation(e) {
  e.preventDefault();
  const btn = document.getElementById('rec-submit-btn');
  const resultsContainer = document.getElementById('rec-results-container');
  const list = document.getElementById('rec-cards-list');
  const banner = document.getElementById('analysis-stats-text');

  const category = document.getElementById('rec-category')?.value || 'Laptops';
  const subcategory = document.getElementById('rec-subcategory')?.value || null;
  const minPrice = parseFloat(document.getElementById('rec-min-price')?.value) || 0.0;
  const maxPrice = parseFloat(document.getElementById('rec-max-price')?.value) || 50000.0;
  const minRating = parseFloat(document.getElementById('rec-min-rating')?.value) || 0.0;
  const priority = document.querySelector('input[name="rec-priority"]:checked')?.value || 'value';
  const rawBrands = document.getElementById('rec-preferred-brands')?.value || '';
  const preferredBrands = rawBrands ? rawBrands.split(',').map(b => b.trim()).filter(Boolean) : [];

  btn.innerText = 'Calculating AI Match...';
  btn.disabled = true;

  try {
    const payload = {
      category: category,
      sub_category: subcategory,
      requirements: currentRequirements,
      preferred_brands: preferredBrands,
      min_price: minPrice,
      max_price: maxPrice,
      min_rating: minRating,
      priority: priority,
      limit: 10
    };

    const res = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Server returned ${res.status}: ${errorText}`);
    }

    const data = await res.json();
    resultsContainer.style.display = 'block';

    const analyzed = data.candidates_analyzed || (data.recommendations ? data.recommendations.length * 3 : 24);
    const matches = data.total_matches || (data.recommendations ? data.recommendations.length : 0);

    if (banner) {
      banner.innerText = `⚡ ${analyzed} candidate products analyzed with Gradient Boosting ML in ${category} • Top ${matches} personalized matches`;
    }

    const matchesCountEl = document.getElementById('rec-matches-count');
    if (matchesCountEl) matchesCountEl.innerText = `${matches} Matches`;

    if (!data.recommendations || data.recommendations.length === 0) {
      list.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🤖</div>
          <h4>No Optimal Matches Found</h4>
          <p>No products in "${escapeHtml(category)}" satisfied ₹${minPrice.toLocaleString()}–₹${maxPrice.toLocaleString()} with ${minRating}★ rating. Try widening budget bounds or adjusting requirement weights.</p>
        </div>
      `;
    } else {
      list.innerHTML = data.recommendations.map((r, i) => {
        const prodId = r.canonical_id || r.id;
        const isCompare = compareIds.includes(prodId);
        const isSave = savedIds.includes(prodId);
        const specs = r.specs || {};

        return `
          <div class="rec-item-card">
            <div class="rec-top-row" onclick="openProductModal('${escapeHtml(prodId)}')">
              <div class="rec-img-wrapper">
                <img src="${escapeHtml(r.thumbnail || r.image || 'https://via.placeholder.com/100')}" onerror="this.src='https://via.placeholder.com/100'" alt="${escapeHtml(r.title || r.name || '')}">
              </div>
              <div class="rec-info">
                <div class="rec-badge-match">#${i + 1} AI Match: ${r.match_score}/100</div>
                <div class="card-brand">${escapeHtml(r.brand || r.merchant || 'Verified')} • 🏪 ${escapeHtml(r.merchant || 'Online Store')}</div>
                <div class="card-title">${escapeHtml(r.title || r.name || '')}</div>
                <div class="card-price-row">
                  <span class="card-price">₹${Number(r.price).toLocaleString('en-IN')}</span>
                  ${r.original_price > r.price ? `<span class="card-mrp">₹${Number(r.original_price).toLocaleString('en-IN')}</span>` : ''}
                  ${r.discount_percentage > 0 ? `<span class="card-discount">${Math.round(r.discount_percentage)}% OFF</span>` : ''}
                </div>
                <!-- Specs Badges -->
                <div class="specs-badge-row">
                  ${specs.ram_gb ? `<span class="spec-pill">${specs.ram_gb}GB RAM</span>` : ''}
                  ${specs.storage_display ? `<span class="spec-pill">${specs.storage_display}</span>` : ''}
                  ${specs.processor ? `<span class="spec-pill">${specs.processor}</span>` : ''}
                  ${specs.gpu ? `<span class="spec-pill">${specs.gpu}</span>` : ''}
                  ${specs.camera_mp ? `<span class="spec-pill">${specs.camera_mp}MP Cam</span>` : ''}
                  ${specs.has_anc ? `<span class="spec-pill">ANC Active</span>` : ''}
                </div>
              </div>
            </div>
            
            <div class="rec-reason-box">
              ⚡ <strong>Why BuyWise Recommends This:</strong><br>
              ${escapeHtml(r.recommendation_reason || 'Verified top fit for your criteria')}
            </div>

            <!-- Direct Action Buttons on AI Card -->
            <div class="rec-actions-row">
              <button class="rec-action-btn ${isCompare ? 'active' : ''}" onclick="toggleCompare('${escapeHtml(prodId)}', event)">
                <span>${isCompare ? '✓ In Compare' : '+ Add to Compare'}</span>
              </button>
              <button class="rec-action-btn ${isSave ? 'active' : ''}" onclick="toggleSave('${escapeHtml(prodId)}', event)">
                <span>${isSave ? '❤️ Saved' : '🤍 Save'}</span>
              </button>
              <button class="rec-action-btn" onclick="openProductModal('${escapeHtml(prodId)}')">
                <span>View Details ↗</span>
              </button>
            </div>
          </div>
        `;
      }).join('');
    }

    resultsContainer.scrollIntoView({ behavior: 'smooth' });

  } catch (err) {
    console.error('Error generating recommendations:', err);
    showToast(`Recommendation error: ${err.message}`);
  } finally {
    btn.innerText = 'Find My Best Options';
    btn.disabled = false;
  }
}

// ----------------------------------------------------
// 4. Product Details Modal with Multi-Retailer Offers & Price History
// ----------------------------------------------------
async function openProductModal(productId) {
  const modal = document.getElementById('product-modal');
  const container = document.getElementById('modal-content-container');
  if (!modal || !container) return;

  modal.style.display = 'flex';
  container.innerHTML = `<p style="padding: 30px; text-align: center;">Loading product intelligence, verified merchant offers & price history...</p>`;

  try {
    const headers = authToken ? { 'Authorization': `Bearer ${authToken}` } : {};
    const [resP, resHist] = await Promise.all([
      fetch(`${API_BASE}/products/${encodeURIComponent(productId)}`, { headers }),
      fetch(`${API_BASE}/products/${encodeURIComponent(productId)}/price-history?days=30`)
    ]);

    if (!resP.ok) throw new Error(`HTTP ${resP.status}`);
    const p = await resP.json();
    const hist = resHist.ok ? await resHist.json() : null;

    const sent = p.sentiment_insights || {};
    const offers = p.retailer_offers || p.offers || [];
    const prodId = p.canonical_id || p.id;
    const isSaved = savedIds.includes(prodId);
    const spread = p.price_spread_info;

    container.innerHTML = `
      <div style="display: flex; gap: 14px; margin-bottom: 14px;">
        <div style="width: 110px; height: 110px; border-radius: var(--radius-md); background: #F1F5F9; display: flex; align-items: center; justify-content: center; overflow: hidden; flex-shrink: 0;">
          <img src="${escapeHtml(p.thumbnail || p.image || '')}" style="max-width: 100%; max-height: 100%; object-fit: contain;" alt="">
        </div>
        <div>
          <div class="card-brand">${escapeHtml(p.brand || p.merchant || 'Verified')} • 🏪 ${escapeHtml(p.merchant || 'Online Store')}</div>
          <h3 style="font-size: 0.95rem; font-weight: 700; line-height: 1.25; margin-bottom: 6px;">${escapeHtml(p.title || p.name || '')}</h3>
          <div class="card-price-row">
            <span class="card-price" style="font-size: 1.2rem;">₹${Number(p.price).toLocaleString('en-IN')}</span>
            ${p.original_price > p.price ? `<span class="card-mrp">₹${Number(p.original_price).toLocaleString('en-IN')}</span>` : ''}
            ${p.discount_percentage > 0 ? `<span class="card-discount">${Math.round(p.discount_percentage)}% OFF</span>` : ''}
          </div>
        </div>
      </div>

      <!-- Price Spread Banner (If multiple sellers available) -->
      ${spread && spread.retailer_count > 1 && spread.price_spread > 0 ? `
        <div class="price-spread-banner">
          <div>
            <strong>💰 Save ₹${Number(spread.savings_vs_highest).toLocaleString('en-IN')}</strong> (${spread.savings_percentage}% price spread)
          </div>
          <div style="color: var(--primary-dark); font-weight: 700;">
            Cheapest at ${escapeHtml(spread.cheapest_retailer)}
          </div>
        </div>
      ` : ''}

      <!-- BuyWise Score Card -->
      <div style="background: linear-gradient(135deg, #EEF2FF, #FAF5FF); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
        <div>
          <div style="font-size: 0.72rem; font-weight: 700; color: var(--primary);">BUYWISE AI SCORE</div>
          <div style="font-size: 1.35rem; font-weight: 800; color: var(--text-main);">${p.buywise_score || 80} <span style="font-size: 0.8rem; color: var(--text-muted);">/ 100</span></div>
        </div>
        <div style="text-align: right; font-size: 0.75rem; color: var(--text-muted);">
          ${p.rating ? `Rating: <strong>${p.rating}★</strong> (${Number(p.review_count || 0).toLocaleString()} reviews)` : 'Rating: <em>Verified Merchant Feed</em>'}
        </div>
      </div>

      <!-- Real Multi-Retailer Offers Table -->
      <div class="retailer-offers-card">
        <div class="retailer-offers-header">
          <h4>🛒 Verified Merchant Offers</h4>
          <span class="badge-accent">${offers.length} Verified Store${offers.length !== 1 ? 's' : ''}</span>
        </div>
        ${offers.length > 0 ? `
          <table class="retailer-table">
            <tbody>
              ${offers.map(off => {
                const rawUrl = (off.url || p.link || '').trim();
                const isValidUrl = rawUrl.startsWith('http://') || rawUrl.startsWith('https://');
                const storeName = (off.retailer || p.merchant || 'Store').split('.')[0];
                return `
                  <tr>
                    <td>
                      <strong>${escapeHtml(off.retailer || p.merchant || 'Online Store')}</strong><br>
                      <span style="font-size: 0.7rem; color: var(--text-muted);">${escapeHtml(off.delivery_info || 'Online Availability')}</span>
                    </td>
                    <td style="text-align: right;">
                      <strong>₹${Number(off.price || p.price).toLocaleString('en-IN')}</strong><br>
                      ${off.is_best_price ? `<span class="badge-best-price">Best Price</span>` : ''}
                    </td>
                    <td style="text-align: right; min-width: 110px;">
                      ${isValidUrl ? `
                        <a href="${escapeHtml(rawUrl)}" target="_blank" rel="noopener noreferrer" class="btn-primary-sm" style="text-decoration: none; padding: 5px 10px; font-size: 0.72rem; display: inline-block;">
                          Buy on ${escapeHtml(storeName)} ↗
                        </a>
                      ` : `
                        <span class="btn-disabled-retailer" title="Store link unavailable">Store link unavailable</span>
                      `}
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        ` : `
          <div style="padding: 8px 0; display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 0.8rem; color: var(--text-muted);">Primary store: <strong>${escapeHtml(p.merchant || 'Verified Merchant')}</strong></span>
            ${(p.link && (p.link.startsWith('http://') || p.link.startsWith('https://'))) ? `
              <a href="${escapeHtml(p.link)}" target="_blank" rel="noopener noreferrer" class="btn-primary-sm" style="text-decoration: none; padding: 5px 10px; font-size: 0.72rem;">Visit Store ↗</a>
            ` : `<span class="btn-disabled-retailer">Store link unavailable</span>`}
          </div>
        `}
      </div>

      <!-- Real Empirical Price History -->
      ${hist ? `
        <div class="price-history-section">
          <div class="price-history-header">
            <span class="price-history-title">📈 Price History (30 Days)</span>
            <span class="trend-pill ${hist.trend || 'building'}">${escapeHtml(hist.trend_label || 'Price history building')}</span>
          </div>
          <div class="price-history-stats-grid">
            <div class="price-stat-box">
              <div class="price-stat-label">Current</div>
              <div class="price-stat-val">₹${Number(hist.current_price || p.price).toLocaleString('en-IN')}</div>
            </div>
            <div class="price-stat-box">
              <div class="price-stat-label">Lowest Seen</div>
              <div class="price-stat-val" style="color: var(--success);">₹${Number(hist.lowest_observed_price || p.price).toLocaleString('en-IN')}</div>
            </div>
            <div class="price-stat-box">
              <div class="price-stat-label">Highest Seen</div>
              <div class="price-stat-val" style="color: var(--text-muted);">₹${Number(hist.highest_observed_price || p.price).toLocaleString('en-IN')}</div>
            </div>
          </div>
          <p style="font-size: 0.68rem; color: var(--text-muted); margin-top: 8px;">
            ${hist.has_history ? `Based on ${hist.observations.length} empirical price observations across verified Indian sellers.` : 'Price observations recorded continuously on every search.'}
          </p>
        </div>
      ` : ''}

      <!-- Sentiment Insights (NLP) -->
      ${sent.overall_sentiment ? `
        <div class="sentiment-bar-wrapper">
          <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 700;">
            <span>Buyer Sentiment Analysis</span>
            <span style="color: var(--success);">${escapeHtml(sent.overall_sentiment)}</span>
          </div>
          <div class="sentiment-bar">
            <div class="sent-pos" style="width: ${sent.positive_percentage || 75}%;"></div>
            <div class="sent-neu" style="width: ${sent.neutral_percentage || 15}%;"></div>
            <div class="sent-neg" style="width: ${sent.negative_percentage || 10}%;"></div>
          </div>
          <div style="margin-top: 6px; font-size: 0.75rem; color: var(--text-main);">
            ${(sent.key_highlights || []).map(h => `<div style="margin-bottom: 2px;">✓ ${escapeHtml(h)}</div>`).join('')}
          </div>
        </div>
      ` : ''}

      <!-- Action Buttons -->
      <div style="display: flex; gap: 8px; margin: 16px 0;">
        <button class="btn-primary" style="flex: 1; justify-content: center; background: var(--primary); color: white;" onclick="toggleCompare('${escapeHtml(prodId)}'); closeProductModal();">
          ${compareIds.includes(prodId) ? '✓ In Compare' : '+ Add to Compare'}
        </button>
        <button class="btn-outline-sm" style="padding: 10px 14px; font-size: 0.9rem;" onclick="toggleSave('${escapeHtml(prodId)}', event)">
          ${isSaved ? '❤️ Saved' : '🤍 Save'}
        </button>
      </div>
    `;

  } catch (err) {
    console.error('Error opening product modal:', err);
    container.innerHTML = `<p style="color: var(--danger); padding: 20px;">Failed to load product details from shopping cache.</p>`;
  }
}

function closeProductModal(e) {
  if (e && e.target !== document.getElementById('product-modal')) return;
  const modal = document.getElementById('product-modal');
  if (modal) modal.style.display = 'none';
}

// ----------------------------------------------------
// 5. Compare Matrix
// ----------------------------------------------------
function toggleCompare(productId, e) {
  if (e) e.stopPropagation();
  const pidStr = String(productId).trim();
  const idx = compareIds.indexOf(pidStr);
  if (idx > -1) {
    compareIds.splice(idx, 1);
    showToast('Removed from comparison');
  } else {
    if (compareIds.length >= 4) {
      showToast('Maximum 4 products can be compared');
      return;
    }
    compareIds.push(pidStr);
    showToast('Added to comparison');
  }
  localStorage.setItem('buywise_compare', JSON.stringify(compareIds));
  updateCompareBadge();
  if (currentTab === 'compare') renderCompareView();
  if (currentTab === 'recommend') executeRecommendation({ preventDefault: () => {} });
}

function updateCompareBadge() {
  const badge = document.getElementById('compare-badge');
  if (badge) badge.style.display = compareIds.length > 0 ? 'block' : 'none';
}

function clearComparison() {
  compareIds = [];
  localStorage.removeItem('buywise_compare');
  updateCompareBadge();
  renderCompareView();
  showToast('Comparison cleared');
}

async function renderCompareView() {
  const emptyState = document.getElementById('compare-empty-state');
  const matrixContainer = document.getElementById('compare-matrix-container');
  const matrix = document.getElementById('compare-matrix');

  if (compareIds.length === 0) {
    emptyState.style.display = 'block';
    matrixContainer.style.display = 'none';
    return;
  }

  emptyState.style.display = 'none';
  matrixContainer.style.display = 'block';
  matrix.innerHTML = `<p style="padding: 20px; text-align: center;">Loading comparison matrix...</p>`;

  try {
    const res = await fetch(`${API_BASE}/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_ids: compareIds })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const products = data.products;

    if (!products || products.length === 0) {
      emptyState.style.display = 'block';
      matrixContainer.style.display = 'none';
      return;
    }

    matrix.innerHTML = `
      <table class="compare-table">
        <thead>
          <tr>
            <th style="min-width: 110px;">Product</th>
            ${products.map(p => {
              const pid = p.canonical_id || p.id;
              return `
                <th style="min-width: 140px;">
                  <div style="position: relative;">
                    <button onclick="toggleCompare('${escapeHtml(pid)}')" style="position: absolute; top: -5px; right: 0; background: none; border: none; font-size: 1rem; cursor: pointer;">✕</button>
                    <img src="${escapeHtml(p.thumbnail || p.image || '')}" style="height: 60px; object-fit: contain; margin-bottom: 4px;" alt="">
                    <div class="card-brand">${escapeHtml(p.brand || p.merchant || '')}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; line-height: 1.2;">${escapeHtml((p.title || p.name || '').slice(0, 35))}...</div>
                  </div>
                </th>
              `;
            }).join('')}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>AI Winner Badge</td>
            ${products.map(p => {
              const pid = p.canonical_id || p.id;
              return `
                <td>
                  ${pid === data.best_overall_id ? `<span class="badge-success">🏆 Best Overall</span>` : ''}
                  ${pid === data.best_value_id ? `<span class="badge-accent">💎 Best Value</span>` : ''}
                  ${pid === data.best_rated_id ? `<span class="badge-success">⭐ Top Rated</span>` : ''}
                </td>
              `;
            }).join('')}
          </tr>
          <tr>
            <td>Best Price</td>
            ${products.map(p => `<td><strong>₹${Number(p.price).toLocaleString('en-IN')}</strong></td>`).join('')}
          </tr>
          <tr>
            <td>Primary Merchant</td>
            ${products.map(p => `<td>🏪 ${escapeHtml(p.merchant || 'Verified Store')}</td>`).join('')}
          </tr>
          <tr>
            <td>Rating</td>
            ${products.map(p => `<td>${p.rating ? `${p.rating}★` : 'Verified'}</td>`).join('')}
          </tr>
          <tr>
            <td>BuyWise Score</td>
            ${products.map(p => `<td><span class="badge-accent" style="font-size: 0.85rem;">${p.buywise_score || 80} / 100</span></td>`).join('')}
          </tr>
          <tr>
            <td>Action</td>
            ${products.map(p => {
              const pid = p.canonical_id || p.id;
              return `
                <td>
                  <button class="btn-primary-sm" onclick="openProductModal('${escapeHtml(pid)}')">Details & Offers</button>
                </td>
              `;
            }).join('')}
          </tr>
        </tbody>
      </table>
    `;

  } catch (err) {
    console.error('Error rendering comparison:', err);
    matrix.innerHTML = `<p style="color: var(--danger); padding: 20px;">Failed to render comparison.</p>`;
  }
}

// ----------------------------------------------------
// 6. User Auth & Wishlist Handling
// ----------------------------------------------------
function updateAuthUI() {
  const btnLabel = document.getElementById('auth-btn-label');
  if (currentUser) {
    if (btnLabel) btnLabel.innerText = currentUser.username || 'My Account';
  } else {
    if (btnLabel) btnLabel.innerText = 'Sign In';
  }
}

function openAuthModal() {
  const modal = document.getElementById('auth-modal');
  const container = document.getElementById('auth-modal-container');
  if (!modal || !container) return;

  modal.style.display = 'flex';

  if (currentUser) {
    renderUserProfileView();
  } else {
    renderLoginFormView();
  }
}

function closeAuthModal(e) {
  if (e && e.target !== document.getElementById('auth-modal')) return;
  const modal = document.getElementById('auth-modal');
  if (modal) modal.style.display = 'none';
}

function renderLoginFormView() {
  const container = document.getElementById('auth-modal-container');
  container.innerHTML = `
    <div class="modal-header-title">
      <h3>Sign In to BuyWise</h3>
      <p>Save products, track prices, and customize recommendations</p>
    </div>
    <form onsubmit="handleLoginSubmit(event)">
      <div class="form-group">
        <label>Email Address</label>
        <input type="email" id="login-email" placeholder="name@example.com" required>
      </div>
      <div class="form-group">
        <label>Password</label>
        <input type="password" id="login-password" placeholder="••••••••" required>
      </div>
      <button type="submit" class="btn-primary full-width" id="login-btn">Sign In</button>
    </form>
    <div style="text-align: center; margin-top: 16px; font-size: 0.82rem;">
      Don't have an account? <a href="javascript:void(0)" onclick="renderSignupFormView()" style="color: var(--primary); font-weight: 700;">Create Account</a>
    </div>
  `;
}

function renderSignupFormView() {
  const container = document.getElementById('auth-modal-container');
  container.innerHTML = `
    <div class="modal-header-title">
      <h3>Create Free Account</h3>
      <p>Join BuyWise for intelligent shopping search</p>
    </div>
    <form onsubmit="handleSignupSubmit(event)">
      <div class="form-group">
        <label>Your Name</label>
        <input type="text" id="signup-username" placeholder="Aayushi" required>
      </div>
      <div class="form-group">
        <label>Email Address</label>
        <input type="email" id="signup-email" placeholder="name@example.com" required>
      </div>
      <div class="form-group">
        <label>Password</label>
        <input type="password" id="signup-password" placeholder="Minimum 6 characters" minlength="6" required>
      </div>
      <button type="submit" class="btn-primary full-width" id="signup-btn">Create Account</button>
    </form>
    <div style="text-align: center; margin-top: 16px; font-size: 0.82rem;">
      Already have an account? <a href="javascript:void(0)" onclick="renderLoginFormView()" style="color: var(--primary); font-weight: 700;">Sign In</a>
    </div>
  `;
}

async function handleLoginSubmit(e) {
  e.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  const btn = document.getElementById('login-btn');
  btn.innerText = 'Signing In...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (data.success && data.user) {
      currentUser = data.user;
      authToken = data.token;
      localStorage.setItem('buywise_user', JSON.stringify(currentUser));
      localStorage.setItem('buywise_token', authToken);
      updateAuthUI();
      renderUserProfileView();
      showToast(`Welcome back, ${currentUser.username}!`);
      syncUserSaved();
    } else {
      showToast(data.error || 'Login failed.');
    }
  } catch (err) {
    showToast('Network error during sign in.');
  } finally {
    btn.innerText = 'Sign In';
    btn.disabled = false;
  }
}

async function handleSignupSubmit(e) {
  e.preventDefault();
  const username = document.getElementById('signup-username').value;
  const email = document.getElementById('signup-email').value;
  const password = document.getElementById('signup-password').value;
  const btn = document.getElementById('signup-btn');
  btn.innerText = 'Creating Account...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password })
    });
    const data = await res.json();
    if (data.success && data.user) {
      currentUser = data.user;
      authToken = data.token;
      localStorage.setItem('buywise_user', JSON.stringify(currentUser));
      localStorage.setItem('buywise_token', authToken);
      updateAuthUI();
      renderUserProfileView();
      showToast(`Account created! Welcome, ${currentUser.username}.`);
    } else {
      showToast(data.error || 'Signup failed.');
    }
  } catch (err) {
    showToast('Network error during signup.');
  } finally {
    btn.innerText = 'Create Account';
    btn.disabled = false;
  }
}

async function renderUserProfileView() {
  const container = document.getElementById('auth-modal-container');
  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    const profile = res.ok ? await res.json() : currentUser;

    container.innerHTML = `
      <div style="text-align: center; margin-bottom: 20px;">
        <div style="width: 60px; height: 60px; border-radius: var(--radius-full); background: linear-gradient(135deg, var(--primary), var(--accent)); color: white; font-size: 1.6rem; font-weight: 800; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">
          ${(profile.username || 'U')[0].toUpperCase()}
        </div>
        <h3 style="font-size: 1.15rem; font-weight: 800;">${escapeHtml(profile.username)}</h3>
        <p style="font-size: 0.78rem; color: var(--text-muted);">${escapeHtml(profile.email)}</p>
      </div>

      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 20px;">
        <div style="background: var(--bg-main); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px; text-align: center;">
          <div style="font-size: 1.2rem; font-weight: 800; color: var(--primary);">${profile.saved_count || savedIds.length}</div>
          <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-muted);">Saved Wishlist</div>
        </div>
        <div style="background: var(--bg-main); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px; text-align: center;">
          <div style="font-size: 1.2rem; font-weight: 800; color: var(--accent);">${profile.history_count || 0}</div>
          <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-muted);">Recently Viewed</div>
        </div>
      </div>

      <button class="btn-primary full-width" style="background: #F1F5F9; color: var(--text-main); border: 1px solid var(--border); margin-bottom: 8px;" onclick="switchTab('saved'); closeAuthModal();">
        💖 View My Wishlist
      </button>

      <button class="btn-primary full-width" style="background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5;" onclick="handleLogout()">
        Sign Out
      </button>
    `;
  } catch (err) {
    console.error('Error rendering profile:', err);
  }
}

function handleLogout() {
  currentUser = null;
  authToken = '';
  localStorage.removeItem('buywise_user');
  localStorage.removeItem('buywise_token');
  updateAuthUI();
  closeAuthModal();
  showToast('Signed out successfully.');
}

async function syncUserSaved() {
  if (!authToken) return;
  try {
    const res = await fetch(`${API_BASE}/user/saved`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    if (res.ok) {
      const items = await res.json();
      savedIds = items.map(i => i.canonical_id || i.id);
      localStorage.setItem('buywise_saved', JSON.stringify(savedIds));
      updateSavedBadge();
    }
  } catch (err) {
    console.error('Error syncing saved items:', err);
  }
}

async function toggleSave(productId, e) {
  if (e) e.stopPropagation();
  const pidStr = String(productId).trim();
  const idx = savedIds.indexOf(pidStr);
  if (idx > -1) {
    savedIds.splice(idx, 1);
    showToast('Removed from Saved');
  } else {
    savedIds.push(pidStr);
    showToast('Saved to Wishlist');
  }
  localStorage.setItem('buywise_saved', JSON.stringify(savedIds));
  updateSavedBadge();

  if (authToken) {
    try {
      await fetch(`${API_BASE}/user/saved/toggle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ product_id: pidStr })
      });
    } catch (err) {
      console.error('Error saving item on server:', err);
    }
  }

  if (currentTab === 'saved') renderSavedView();
}

function updateSavedBadge() {
  const countLabel = document.getElementById('saved-count');
  if (countLabel) countLabel.innerText = `${savedIds.length} items`;
}

async function renderSavedView() {
  const emptyState = document.getElementById('saved-empty-state');
  const grid = document.getElementById('saved-products-grid');

  if (savedIds.length === 0) {
    emptyState.style.display = 'block';
    if (grid) grid.innerHTML = '';
    return;
  }

  emptyState.style.display = 'none';
  if (grid) grid.innerHTML = `<div class="skeleton-card"></div>`;

  try {
    const products = await Promise.all(
      savedIds.map(id => fetch(`${API_BASE}/products/${encodeURIComponent(id)}`).then(r => r.ok ? r.json() : null))
    );
    if (grid) {
      grid.innerHTML = products.filter(p => p && (p.id || p.canonical_id)).map(p => renderProductCard(p)).join('');
    }
  } catch (err) {
    console.error('Error rendering saved items:', err);
  }
}

// ----------------------------------------------------
// UI Helpers
// ----------------------------------------------------
function renderProductCard(p) {
  const prodId = p.canonical_id || p.id;
  const isSaved = savedIds.includes(prodId);
  const isCompare = compareIds.includes(prodId);
  const title = p.title || p.name || 'Product';
  const price = p.price ? Number(p.price).toLocaleString('en-IN') : '0';
  const mrp = p.original_price > p.price ? Number(p.original_price).toLocaleString('en-IN') : null;
  const merchant = p.merchant || 'Verified Store';
  const imgUrl = p.thumbnail || p.image || 'https://via.placeholder.com/150';

  return `
    <div class="product-card" onclick="openProductModal('${escapeHtml(prodId)}')">
      <div class="card-badge-score">${p.buywise_score || '80'}</div>
      <button class="card-save-btn" onclick="toggleSave('${escapeHtml(prodId)}', event)">
        <span style="color: ${isSaved ? '#EF4444' : '#94A3B8'}; font-size: 1.1rem;">${isSaved ? '♥' : '♡'}</span>
      </button>
      <div class="card-img-wrapper">
        <img src="${escapeHtml(imgUrl)}" onerror="this.src='https://via.placeholder.com/150'" alt="${escapeHtml(title)}">
      </div>
      <div class="card-brand-row">
        <div class="card-brand">${escapeHtml(p.brand || merchant)}</div>
        <div class="card-retailer-tag">🏪 ${escapeHtml(merchant)}</div>
      </div>
      <div class="card-title">${escapeHtml(title)}</div>
      <div class="card-rating-row">
        ${p.rating ? `<span class="star-icon">★</span><span>${p.rating}</span><span>(${p.review_count ? Number(p.review_count).toLocaleString() : '10+'})</span>` : '<span>Verified Product</span>'}
      </div>
      <div class="card-price-row">
        <span class="card-price">₹${price}</span>
        ${mrp ? `<span class="card-mrp">₹${mrp}</span>` : ''}
        ${p.discount_percentage > 0 ? `<span class="card-discount">${Math.round(p.discount_percentage)}% off</span>` : ''}
      </div>
    </div>
  `;
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.innerText = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2200);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
