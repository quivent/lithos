(function() {
  function openSearch(e) {
    if (e) e.preventDefault();
    var overlay = document.getElementById('search-overlay');
    overlay.style.display = 'flex';
    var input = document.getElementById('search-input');
    input.value = '';
    document.getElementById('search-results').innerHTML = '';
    setTimeout(function() { input.focus(); }, 50);
  }

  function closeSearch(e) {
    if (e && e.target && e.target.id !== 'search-overlay') return;
    var overlay = document.getElementById('search-overlay');
    overlay.style.display = 'none';
    document.getElementById('search-results').innerHTML = '';
    document.getElementById('search-input').value = '';
  }

  function doSearch(query) {
    var results = document.getElementById('search-results');
    if (!query || query.length < 2) {
      results.innerHTML = '';
      return;
    }
    var q = query.toLowerCase();
    var terms = q.split(/\s+/).filter(function(t) { return t.length > 0; });
    var scored = [];

    for (var i = 0; i < SEARCH_INDEX.length; i++) {
      var entry = SEARCH_INDEX[i];
      var score = 0;
      var matchedSection = '';
      var titleLower = entry.title.toLowerCase();
      var termsLower = (entry.terms || '').toLowerCase();

      for (var t = 0; t < terms.length; t++) {
        var term = terms[t];
        if (titleLower.indexOf(term) !== -1) score += 10;
        if (termsLower.indexOf(term) !== -1) score += 3;
        for (var s = 0; s < entry.sections.length; s++) {
          if (entry.sections[s].toLowerCase().indexOf(term) !== -1) {
            score += 5;
            if (!matchedSection) matchedSection = entry.sections[s];
          }
        }
      }

      if (score > 0) {
        scored.push({ entry: entry, score: score, section: matchedSection });
      }
    }

    scored.sort(function(a, b) { return b.score - a.score; });

    if (scored.length === 0) {
      results.innerHTML = '<div style="padding:1rem;color:#555568;font-size:0.85rem">No results found.</div>';
      return;
    }

    var html = '';
    var max = Math.min(scored.length, 12);
    for (var r = 0; r < max; r++) {
      var item = scored[r];
      var sectionHtml = item.section
        ? '<div style="color:#888898;font-size:0.78rem;margin-top:0.15rem">' + escapeHtml(item.section) + '</div>'
        : '';
      var snippet = buildSnippet(item.entry, terms);
      html += '<a href="' + item.entry.url + '" style="display:block;padding:0.7rem 1rem;background:#10101a;border:1px solid #1a1a28;border-radius:3px;margin-bottom:0.35rem;text-decoration:none;transition:border-color 0.15s"'
        + ' onmouseenter="this.style.borderColor=\'#d4a053\'" onmouseleave="this.style.borderColor=\'#1a1a28\'">'
        + '<div style="color:#d4a053;font-size:0.88rem;font-weight:500">' + escapeHtml(item.entry.title) + '</div>'
        + sectionHtml
        + '<div style="color:#888898;font-size:0.76rem;margin-top:0.25rem">' + snippet + '</div>'
        + '</a>';
    }
    results.innerHTML = html;
  }

  function buildSnippet(entry, terms) {
    var parts = (entry.terms || '').split(/\s+/);
    var matched = [];
    for (var i = 0; i < parts.length && matched.length < 8; i++) {
      for (var t = 0; t < terms.length; t++) {
        if (parts[i].indexOf(terms[t]) !== -1 && matched.indexOf(parts[i]) === -1) {
          matched.push(parts[i]);
        }
      }
    }
    if (matched.length > 0) return matched.join(' \u00b7 ');
    return entry.sections.slice(0, 3).join(' \u00b7 ');
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      openSearch(null);
    }
    if (e.key === 'Escape') {
      var overlay = document.getElementById('search-overlay');
      if (overlay && overlay.style.display === 'flex') {
        overlay.style.display = 'none';
        document.getElementById('search-results').innerHTML = '';
        document.getElementById('search-input').value = '';
      }
    }
  });

  // Expose globally
  window.openSearch = openSearch;
  window.closeSearch = closeSearch;
  window.doSearch = doSearch;
})();
