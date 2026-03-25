/* global Plotly */
(function () {
  'use strict';

  const PCAPlayground = (() => {
    function cloneMatrix(matrix) {
      return matrix.map((row) => row.slice());
    }

    function zeros(rows, cols) {
      return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
    }

    function identity(n) {
      const out = zeros(n, n);
      for (let i = 0; i < n; i += 1) {
        out[i][i] = 1;
      }
      return out;
    }

    function diagMatrix(values) {
      const out = zeros(values.length, values.length);
      for (let i = 0; i < values.length; i += 1) {
        out[i][i] = values[i];
      }
      return out;
    }

    function symmetrize(matrix) {
      const n = matrix.length;
      const out = zeros(n, n);
      for (let i = 0; i < n; i += 1) {
        for (let j = i; j < n; j += 1) {
          const value = 0.5 * (matrix[i][j] + matrix[j][i]);
          out[i][j] = value;
          out[j][i] = value;
        }
      }
      return out;
    }

    function transpose(matrix) {
      const rows = matrix.length;
      const cols = matrix[0].length;
      const out = zeros(cols, rows);
      for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
          out[j][i] = matrix[i][j];
        }
      }
      return out;
    }

    function matMul(a, b) {
      const aRows = a.length;
      const aCols = a[0].length;
      const bCols = b[0].length;
      const out = zeros(aRows, bCols);
      for (let i = 0; i < aRows; i += 1) {
        for (let k = 0; k < aCols; k += 1) {
          const aik = a[i][k];
          for (let j = 0; j < bCols; j += 1) {
            out[i][j] += aik * b[k][j];
          }
        }
      }
      return out;
    }

    function addRowVector(matrix, vector) {
      return matrix.map((row) => row.map((value, index) => value + vector[index]));
    }

    function maxAbsOffDiagonal(matrix) {
      let maxValue = 0;
      let p = 0;
      let q = 1;
      const n = matrix.length;
      for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
          const value = Math.abs(matrix[i][j]);
          if (value > maxValue) {
            maxValue = value;
            p = i;
            q = j;
          }
        }
      }
      return { maxValue, p, q };
    }

    function orientEigenvectors(vectors) {
      const n = vectors.length;
      const oriented = cloneMatrix(vectors);
      for (let j = 0; j < n; j += 1) {
        let pivotRow = 0;
        let pivotAbs = Math.abs(oriented[0][j]);
        for (let i = 1; i < n; i += 1) {
          const candidate = Math.abs(oriented[i][j]);
          if (candidate > pivotAbs) {
            pivotAbs = candidate;
            pivotRow = i;
          }
        }
        if (oriented[pivotRow][j] < 0) {
          for (let i = 0; i < n; i += 1) {
            oriented[i][j] *= -1;
          }
        }
      }
      return oriented;
    }

    function jacobiEigen(matrix, tolerance = 1e-12, maxIterations = 100) {
      const n = matrix.length;
      const a = cloneMatrix(matrix);
      let v = identity(n);
      const totalIterations = maxIterations * n * n;

      for (let iter = 0; iter < totalIterations; iter += 1) {
        const { maxValue, p, q } = maxAbsOffDiagonal(a);
        if (maxValue < tolerance) {
          break;
        }

        const app = a[p][p];
        const aqq = a[q][q];
        const apq = a[p][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);

        for (let i = 0; i < n; i += 1) {
          if (i !== p && i !== q) {
            const aip = a[i][p];
            const aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[p][i] = a[i][p];
            a[i][q] = s * aip + c * aiq;
            a[q][i] = a[i][q];
          }
        }

        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p][q] = 0;
        a[q][p] = 0;

        for (let i = 0; i < n; i += 1) {
          const vip = v[i][p];
          const viq = v[i][q];
          v[i][p] = c * vip - s * viq;
          v[i][q] = s * vip + c * viq;
        }
      }

      const pairs = [];
      for (let i = 0; i < n; i += 1) {
        pairs.push({
          value: a[i][i],
          vector: v.map((row) => row[i])
        });
      }
      pairs.sort((left, right) => right.value - left.value);

      const values = pairs.map((entry) => entry.value);
      const vectors = zeros(n, n);
      for (let j = 0; j < n; j += 1) {
        for (let i = 0; i < n; i += 1) {
          vectors[i][j] = pairs[j].vector[i];
        }
      }

      return {
        values,
        vectors: orientEigenvectors(vectors)
      };
    }

    function columnMeans(data) {
      const rows = data.length;
      const cols = data[0].length;
      const means = Array.from({ length: cols }, () => 0);
      for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
          means[j] += data[i][j];
        }
      }
      for (let j = 0; j < cols; j += 1) {
        means[j] /= rows;
      }
      return means;
    }

    function columnSds(data, means = null) {
      const rows = data.length;
      const cols = data[0].length;
      const mu = means || columnMeans(data);
      const variances = Array.from({ length: cols }, () => 0);
      for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
          const diff = data[i][j] - mu[j];
          variances[j] += diff * diff;
        }
      }
      for (let j = 0; j < cols; j += 1) {
        variances[j] = rows > 1 ? Math.sqrt(variances[j] / (rows - 1)) : 0;
      }
      return variances;
    }

    function centerScale(data, shouldScale = false) {
      const means = columnMeans(data);
      const sds = columnSds(data, means).map((value) => (value > 0 ? value : 1));
      const transformed = data.map((row) => row.map((value, index) => {
        const centered = value - means[index];
        return shouldScale ? centered / sds[index] : centered;
      }));
      return { data: transformed, means, sds };
    }

    function covarianceMatrix(centeredData) {
      const rows = centeredData.length;
      const cols = centeredData[0].length;
      const out = zeros(cols, cols);
      const denom = Math.max(rows - 1, 1);
      for (let i = 0; i < cols; i += 1) {
        for (let j = i; j < cols; j += 1) {
          let sum = 0;
          for (let r = 0; r < rows; r += 1) {
            sum += centeredData[r][i] * centeredData[r][j];
          }
          out[i][j] = sum / denom;
          out[j][i] = out[i][j];
        }
      }
      return out;
    }

    function correlationMatrix(data) {
      return covarianceMatrix(centerScale(data, true).data);
    }

    function mulberry32(seed) {
      let t = seed >>> 0;
      return function next() {
        t += 0x6D2B79F5;
        let x = t;
        x = Math.imul(x ^ (x >>> 15), x | 1);
        x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
        return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
      };
    }

    function normalGenerator(seed) {
      const random = mulberry32(seed);
      let spare = null;
      return function draw() {
        if (spare !== null) {
          const value = spare;
          spare = null;
          return value;
        }
        let u = 0;
        let v = 0;
        while (u === 0) {
          u = random();
        }
        while (v === 0) {
          v = random();
        }
        const radius = Math.sqrt(-2 * Math.log(u));
        const theta = 2 * Math.PI * v;
        spare = radius * Math.sin(theta);
        return radius * Math.cos(theta);
      };
    }

    function validateSigma(sigma) {
      const symCheck = sigma.every((row, i) => row.every((value, j) => Math.abs(value - sigma[j][i]) < 1e-9));
      if (!symCheck) {
        return { ok: false, code: 'nonsymmetric', message: 'Sigma must be symmetric.' };
      }
      for (let i = 0; i < sigma.length; i += 1) {
        if (!(sigma[i][i] > 0)) {
          return { ok: false, code: 'nonpositive-variance', message: 'Variances on the diagonal must be positive.' };
        }
      }
      const eigen = jacobiEigen(sigma);
      const minimum = Math.min.apply(null, eigen.values);
      if (minimum < -1e-8) {
        return {
          ok: false,
          code: 'not-psd',
          message: 'Sigma is not positive semidefinite.',
          minEigenvalue: minimum,
          eigenvalues: eigen.values,
          eigenvectors: eigen.vectors
        };
      }
      return {
        ok: true,
        code: 'ok',
        eigenvalues: eigen.values,
        eigenvectors: eigen.vectors,
        minEigenvalue: minimum
      };
    }

    function repairSigma(sigma, eigenFloor = 1e-8) {
      const symmetric = symmetrize(sigma);
      const originalDiag = symmetric.map((row, index) => Math.max(row[index], eigenFloor));
      const eigen = jacobiEigen(symmetric);
      const clippedEigenvalues = eigen.values.map((value) => Math.max(value, eigenFloor));
      let repaired = matMul(
        matMul(eigen.vectors, diagMatrix(clippedEigenvalues)),
        transpose(eigen.vectors)
      );
      repaired = symmetrize(repaired);

      const repairedDiag = repaired.map((row, index) => Math.max(row[index], eigenFloor));
      const scaleFactors = originalDiag.map((value, index) => Math.sqrt(value / repairedDiag[index]));
      const scaleMatrix = diagMatrix(scaleFactors);
      repaired = matMul(matMul(scaleMatrix, repaired), scaleMatrix);
      repaired = symmetrize(repaired);

      let validation = validateSigma(repaired);
      let diagonalLift = 0;
      if (!validation.ok && validation.code === 'not-psd') {
        diagonalLift = Math.abs(validation.minEigenvalue) + eigenFloor;
        for (let i = 0; i < repaired.length; i += 1) {
          repaired[i][i] += diagonalLift;
        }
        repaired = symmetrize(repaired);
        validation = validateSigma(repaired);
      }

      return {
        ok: validation.ok,
        sigma: repaired,
        minOriginalEigenvalue: Math.min.apply(null, eigen.values),
        originalEigenvalues: eigen.values,
        clippedEigenvalues,
        diagonalLift
      };
    }

    function generateMVN({ n, means, sigma, seed }) {
      const validation = validateSigma(sigma);
      if (!validation.ok) {
        throw new Error(validation.message);
      }
      const p = means.length;
      const sqrtEigen = validation.eigenvalues.map((value) => Math.sqrt(Math.max(value, 0)));
      const transform = matMul(validation.eigenvectors, diagMatrix(sqrtEigen));
      const drawNormal = normalGenerator(seed);
      const z = zeros(n, p);
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < p; j += 1) {
          z[i][j] = drawNormal();
        }
      }
      return addRowVector(matMul(z, transpose(transform)), means);
    }

    function pca(data, scaleData) {
      const prepared = centerScale(data, scaleData);
      const covariance = covarianceMatrix(prepared.data);
      const eigen = jacobiEigen(covariance);
      const scores = matMul(prepared.data, eigen.vectors);
      const loadings = matMul(eigen.vectors, diagMatrix(eigen.values.map((value) => Math.sqrt(Math.max(value, 0)))));
      const totalVariance = eigen.values.reduce((acc, value) => acc + value, 0);
      const explained = eigen.values.map((value) => (totalVariance > 0 ? value / totalVariance : 0));
      const cumulative = [];
      explained.reduce((acc, value, index) => {
        const next = acc + value;
        cumulative[index] = next;
        return next;
      }, 0);
      return {
        covariance,
        eigenvalues: eigen.values,
        eigenvectors: eigen.vectors,
        loadings,
        scores,
        explained,
        cumulative,
        center: prepared.means,
        scale: prepared.sds
      };
    }

    function quantile(sortedValues, probability) {
      if (sortedValues.length === 0) {
        return NaN;
      }
      const index = (sortedValues.length - 1) * probability;
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      if (lower === upper) {
        return sortedValues[lower];
      }
      const weight = index - lower;
      return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
    }

    function parallelAnalysis(data, scaleData, seed, iterations = 100) {
      const n = data.length;
      const p = data[0].length;
      const sds = columnSds(data).map((value) => (value > 0 ? value : 1));
      const drawNormal = normalGenerator(seed + 982451653);
      const storage = Array.from({ length: p }, () => []);
      for (let iter = 0; iter < iterations; iter += 1) {
        const sim = zeros(n, p);
        for (let i = 0; i < n; i += 1) {
          for (let j = 0; j < p; j += 1) {
            const value = drawNormal();
            sim[i][j] = scaleData ? value : value * sds[j];
          }
        }
        const simPca = pca(sim, scaleData);
        for (let k = 0; k < p; k += 1) {
          storage[k].push(simPca.eigenvalues[k]);
        }
      }
      const mean = storage.map((values) => values.reduce((acc, value) => acc + value, 0) / values.length);
      const q95 = storage.map((values) => {
        const sorted = values.slice().sort((left, right) => left - right);
        return quantile(sorted, 0.95);
      });
      return { mean, q95, iterations };
    }

    function sampleIndices(n, maxPoints, seed) {
      if (n <= maxPoints) {
        return Array.from({ length: n }, (_, index) => index);
      }
      const draw = mulberry32(seed + 17);
      const indices = Array.from({ length: n }, (_, index) => index);
      for (let i = n - 1; i > 0; i -= 1) {
        const j = Math.floor(draw() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
      return indices.slice(0, maxPoints).sort((left, right) => left - right);
    }

    return {
      cloneMatrix,
      transpose,
      matMul,
      addRowVector,
      jacobiEigen,
      columnMeans,
      columnSds,
      centerScale,
      covarianceMatrix,
      correlationMatrix,
      validateSigma,
      repairSigma,
      generateMVN,
      pca,
      parallelAnalysis,
      sampleIndices
    };
  })();

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = PCAPlayground;
  }

  if (typeof window === 'undefined') {
    return;
  }

  const state = {
    p: 3,
    n: 250,
    preset: 'factor',
    scale: true,
    seed: 1234,
    means: [0, 0, 0],
    sigma: [],
    lastRun: null
  };

  const elements = {
    variableCount: document.getElementById('variable-count'),
    presetSelect: document.getElementById('preset-select'),
    nSlider: document.getElementById('n-slider'),
    nLabel: document.getElementById('n-label'),
    scaleToggle: document.getElementById('scale-toggle'),
    seedInput: document.getElementById('seed-input'),
    meansGrid: document.getElementById('means-grid'),
    sigmaGrid: document.getElementById('sigma-grid'),
    runButton: document.getElementById('run-button'),
    resetButton: document.getElementById('reset-button'),
    aboutButton: document.getElementById('about-button'),
    closeAbout: document.getElementById('close-about'),
    aboutModal: document.getElementById('about-modal'),
    statusBox: document.getElementById('status-box'),
    plotNote: document.getElementById('plot-note'),
    summaryParallel: document.getElementById('summary-parallel'),
    summaryKaiser: document.getElementById('summary-kaiser'),
    summaryPC1: document.getElementById('summary-pc1'),
    summaryPC12: document.getElementById('summary-pc12'),
    dataPreview: document.getElementById('data-preview'),
    targetMatrix: document.getElementById('target-matrix'),
    observedCovariance: document.getElementById('observed-covariance'),
    observedCorrelation: document.getElementById('observed-correlation'),
    pcaPrintout: document.getElementById('pca-printout'),
    loadingsTable: document.getElementById('loadings-table'),
    scoresPreview: document.getElementById('scores-preview')
  };

  function variableNames() {
    return Array.from({ length: state.p }, (_, index) => `X${index + 1}`);
  }

  function presetSigma(p, presetName) {
    const presets = {
      factor: {
        3: [
          [1.0, 0.7, 0.55],
          [0.7, 1.0, 0.6],
          [0.55, 0.6, 1.0]
        ],
        4: [
          [1.0, 0.72, 0.64, 0.58],
          [0.72, 1.0, 0.66, 0.54],
          [0.64, 0.66, 1.0, 0.6],
          [0.58, 0.54, 0.6, 1.0]
        ]
      },
      block: {
        3: [
          [1.0, 0.75, 0.15],
          [0.75, 1.0, 0.12],
          [0.15, 0.12, 1.0]
        ],
        4: [
          [1.0, 0.8, 0.15, 0.1],
          [0.8, 1.0, 0.12, 0.15],
          [0.15, 0.12, 1.0, 0.72],
          [0.1, 0.15, 0.72, 1.0]
        ]
      },
      contrast: {
        3: [
          [1.0, 0.68, -0.45],
          [0.68, 1.0, -0.5],
          [-0.45, -0.5, 1.0]
        ],
        4: [
          [1.0, 0.68, -0.42, -0.38],
          [0.68, 1.0, -0.48, -0.44],
          [-0.42, -0.48, 1.0, 0.74],
          [-0.38, -0.44, 0.74, 1.0]
        ]
      },
      independent: {
        3: [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ],
        4: [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      }
    };
    return PCAPlayground.cloneMatrix(presets[presetName][String(p)]);
  }

  function defaultMeans(p) {
    return Array.from({ length: p }, () => 0);
  }

  function formatNumber(value, digits = 3) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return '';
    }
    return value.toFixed(digits);
  }

  function setStatus(message = '', tone = 'error') {
    if (!message) {
      elements.statusBox.style.display = 'none';
      elements.statusBox.textContent = '';
      elements.statusBox.style.background = '#fff7ed';
      elements.statusBox.style.color = '#c2410c';
      elements.statusBox.style.borderColor = '#fdba74';
      return;
    }

    const palette = {
      error: {
        background: '#fff7ed',
        color: '#c2410c',
        borderColor: '#fdba74'
      },
      warning: {
        background: '#fef3c7',
        color: '#92400e',
        borderColor: '#fcd34d'
      },
      info: {
        background: '#eff6ff',
        color: '#1d4ed8',
        borderColor: '#93c5fd'
      }
    }[tone] || {
      background: '#fff7ed',
      color: '#c2410c',
      borderColor: '#fdba74'
    };

    elements.statusBox.style.display = 'block';
    elements.statusBox.textContent = message;
    elements.statusBox.style.background = palette.background;
    elements.statusBox.style.color = palette.color;
    elements.statusBox.style.borderColor = palette.borderColor;
  }

  function syncControlState() {
    state.p = Number(elements.variableCount.value);
    state.preset = elements.presetSelect.value;
    state.n = Number(elements.nSlider.value);
    state.scale = elements.scaleToggle.checked;
    state.seed = Math.max(1, Number(elements.seedInput.value || 1234));
    elements.nLabel.textContent = state.n.toLocaleString();
  }

  function buildMeansGrid() {
    elements.meansGrid.innerHTML = '';
    const names = variableNames();
    names.forEach((name, index) => {
      const wrap = document.createElement('label');
      wrap.innerHTML = `
        <span class="muted">${name}</span>
        <input type="number" step="0.1" data-mean-index="${index}" value="${formatNumber(state.means[index], 2)}">
      `;
      elements.meansGrid.appendChild(wrap);
    });

    elements.meansGrid.querySelectorAll('input').forEach((input) => {
      input.addEventListener('input', (event) => {
        const index = Number(event.target.dataset.meanIndex);
        const value = Number(event.target.value);
        state.means[index] = Number.isFinite(value) ? value : 0;
      });
    });
  }

  function buildSigmaGrid() {
    const table = document.createElement('table');
    table.className = 'matrix-table';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headRow.appendChild(document.createElement('th'));
    variableNames().forEach((name) => {
      const th = document.createElement('th');
      th.textContent = name;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (let i = 0; i < state.p; i += 1) {
      const tr = document.createElement('tr');
      const rowHead = document.createElement('th');
      rowHead.textContent = `X${i + 1}`;
      tr.appendChild(rowHead);

      for (let j = 0; j < state.p; j += 1) {
        const td = document.createElement('td');
        const input = document.createElement('input');
        input.type = 'number';
        input.step = '0.01';
        input.className = 'matrix-input';
        if (i < j) {
          td.className = 'readonly-cell';
          input.classList.add('readonly');
          input.readOnly = true;
          input.tabIndex = -1;
          input.value = formatNumber(state.sigma[i][j], 2);
          input.dataset.row = String(i);
          input.dataset.col = String(j);
        } else {
          input.value = formatNumber(state.sigma[i][j], 2);
          input.dataset.row = String(i);
          input.dataset.col = String(j);
          input.addEventListener('input', onSigmaChange);
        }
        td.appendChild(input);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }

    table.appendChild(tbody);
    elements.sigmaGrid.innerHTML = '';
    elements.sigmaGrid.appendChild(table);
  }

  function onSigmaChange(event) {
    const row = Number(event.target.dataset.row);
    const col = Number(event.target.dataset.col);
    let value = Number(event.target.value);
    if (!Number.isFinite(value)) {
      value = 0;
    }
    if (row === col && value <= 0) {
      value = 0.01;
      event.target.value = formatNumber(value, 2);
    }
    state.sigma[row][col] = value;
    state.sigma[col][row] = value;
    const mirror = elements.sigmaGrid.querySelector(`input[data-row="${col}"][data-col="${row}"]`);
    if (mirror && mirror !== event.target) {
      mirror.value = formatNumber(value, 2);
    }
  }

  function applyPreset() {
    state.means = defaultMeans(state.p);
    state.sigma = presetSigma(state.p, state.preset);
    buildMeansGrid();
    buildSigmaGrid();
  }

  function htmlTable(headers, rows, rowNames = null, digits = 3) {
    const table = document.createElement('table');
    table.className = 'data-table';

    const thead = document.createElement('thead');
    const trHead = document.createElement('tr');
    if (rowNames) {
      const blank = document.createElement('th');
      blank.textContent = '';
      trHead.appendChild(blank);
    }
    headers.forEach((header) => {
      const th = document.createElement('th');
      th.textContent = header;
      trHead.appendChild(th);
    });
    thead.appendChild(trHead);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach((row, rowIndex) => {
      const tr = document.createElement('tr');
      if (rowNames) {
        const th = document.createElement('th');
        th.textContent = rowNames[rowIndex];
        tr.appendChild(th);
      }
      row.forEach((value) => {
        const td = document.createElement('td');
        td.textContent = typeof value === 'number' ? formatNumber(value, digits) : String(value);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function renderMatrix(container, matrix, names) {
    container.innerHTML = '';
    container.appendChild(htmlTable(names, matrix, names, 3));
  }

  function renderPreviewTables(run) {
    const names = variableNames();
    const previewRows = run.data.slice(0, Math.min(10, run.data.length));
    elements.dataPreview.innerHTML = '';
    elements.dataPreview.appendChild(htmlTable(names, previewRows, null, 3));

    renderMatrix(elements.targetMatrix, run.sigmaUsed, names);
    renderMatrix(elements.observedCovariance, run.sampleCovariance, names);
    renderMatrix(elements.observedCorrelation, run.sampleCorrelation, names);

    const loadingRows = run.pca.loadings.map((row) => row.slice());
    elements.loadingsTable.innerHTML = '';
    elements.loadingsTable.appendChild(
      htmlTable(
        Array.from({ length: state.p }, (_, index) => `PC${index + 1}`),
        loadingRows,
        names,
        3
      )
    );

    const scoreRows = run.pca.scores.slice(0, Math.min(10, run.pca.scores.length));
    elements.scoresPreview.innerHTML = '';
    elements.scoresPreview.appendChild(
      htmlTable(
        Array.from({ length: state.p }, (_, index) => `PC${index + 1}`),
        scoreRows,
        null,
        3
      )
    );
  }

  function buildPrintout(run) {
    const basis = state.scale ? 'correlation matrix (standardized variables)' : 'covariance matrix (raw scales)';
    const sdLine = run.pca.eigenvalues.map((value) => Math.sqrt(Math.max(value, 0)));
    const lines = [];
    const names = variableNames();

    lines.push('Principal Components Analysis');
    lines.push('');
    lines.push(`Variables: ${names.join(', ')}`);
    lines.push(`Observations: ${state.n}`);
    lines.push(`Input basis: ${basis}`);
    lines.push('');
    lines.push('Component standard deviations');
    lines.push(`  ${sdLine.map((value) => formatNumber(value, 3)).join('  ')}`);
    lines.push('');
    lines.push('Eigenvalues');
    lines.push(`  ${run.pca.eigenvalues.map((value) => formatNumber(value, 3)).join('  ')}`);
    lines.push('');
    lines.push('Proportion of variance');
    lines.push(`  ${run.pca.explained.map((value) => formatNumber(value, 3)).join('  ')}`);
    lines.push('');
    lines.push('Cumulative proportion');
    lines.push(`  ${run.pca.cumulative.map((value) => formatNumber(value, 3)).join('  ')}`);
    lines.push('');
    lines.push('Loadings');
    lines.push(`      ${Array.from({ length: state.p }, (_, index) => `PC${index + 1}`.padStart(8)).join('')}`);
    run.pca.loadings.forEach((row, rowIndex) => {
      const formatted = row.map((value) => formatNumber(value, 3).padStart(8)).join('');
      lines.push(`${names[rowIndex].padEnd(4)} ${formatted}`);
    });
    lines.push('');
    if (run.sigmaRepair) {
      lines.push(`Sigma repaired for this run (min input eigenvalue: ${formatNumber(run.sigmaRepair.minOriginalEigenvalue, 4)}).`);
      lines.push('');
    }
    lines.push(`Parallel analysis (95th percentile): retain ${run.parallel.count} component(s).`);
    if (state.scale) {
      lines.push(`Kaiser rule (> 1): retain ${run.kaiserCount} component(s).`);
    } else {
      lines.push('Kaiser rule (> 1): omitted because eigenvalues are on the covariance scale.');
    }
    lines.push('');
    lines.push('Interpretation note: signs of eigenvectors and loadings may flip without changing the PCA solution.');

    elements.pcaPrintout.textContent = lines.join('\n');
  }

  function renderSummaries(run) {
    elements.summaryParallel.textContent = `${run.parallel.count}`;
    elements.summaryKaiser.textContent = state.scale ? `${run.kaiserCount}` : 'n/a';
    elements.summaryPC1.textContent = `${formatNumber(run.pca.explained[0] * 100, 1)}%`;
    const pc12 = (run.pca.explained[0] + (run.pca.explained[1] || 0)) * 100;
    elements.summaryPC12.textContent = `${formatNumber(pc12, 1)}%`;
  }

  function renderScatterMatrix(run) {
    const names = variableNames();
    const displayData = run.displayData;
    const dimensions = names.map((name, index) => ({
      label: name,
      values: displayData.map((row) => row[index])
    }));

    const trace = {
      type: 'splom',
      dimensions,
      marker: {
        size: displayData.length <= 300 ? 6 : displayData.length <= 1000 ? 4 : 3,
        opacity: 0.58,
        color: '#94a3b8'
      },
      diagonal: { visible: false }
    };

    const layout = {
      margin: { l: 40, r: 20, t: 20, b: 30 },
      dragmode: 'select',
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { family: 'system-ui, sans-serif', size: 12, color: '#111827' }
    };

    Plotly.newPlot('scatter-matrix', [trace], layout, { responsive: true, displayModeBar: false });
  }

  function renderScreePlot(run) {
    const componentPositions = Array.from({ length: state.p }, (_, index) => index + 1);
    const componentLabels = componentPositions.map((index) => `PC${index}`);
    const traces = [
      {
        type: 'bar',
        x: componentPositions,
        y: run.pca.eigenvalues,
        name: 'Observed eigenvalues',
        hovertemplate: 'PC%{x}: %{y:.3f}<extra></extra>'
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: componentPositions,
        y: run.parallel.q95,
        name: 'Parallel analysis threshold',
        hovertemplate: 'PC%{x}: %{y:.3f}<extra></extra>'
      }
    ];

    const shapes = [];
    if (state.scale) {
      shapes.push({
        type: 'line',
        x0: 1,
        x1: state.p,
        y0: 1,
        y1: 1,
        xref: 'x',
        yref: 'y',
        line: { dash: 'dot', color: '#ef4444', width: 2 }
      });
    }

    const layout = {
      margin: { l: 50, r: 20, t: 20, b: 55 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { family: 'system-ui, sans-serif', size: 12, color: '#111827' },
      xaxis: {
        title: 'Component',
        tickmode: 'array',
        tickvals: componentPositions,
        ticktext: componentLabels
      },
      yaxis: { title: 'Eigenvalue' },
      legend: { orientation: 'h', y: 1.12, x: 0 },
      shapes
    };

    Plotly.newPlot('scree-plot', traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderBiplot(run) {
    const scoreSample = run.displayScores;
    const names = variableNames();
    const xScores = scoreSample.map((row) => row[0]);
    const yScores = scoreSample.map((row) => row[1] || 0);
    const vectorColor = '#d97706';
    const pointColor = '#94a3b8';

    const maxScoreX = Math.max(1e-8, Math.max(...xScores.map((value) => Math.abs(value))));
    const maxScoreY = Math.max(1e-8, Math.max(...yScores.map((value) => Math.abs(value))));
    const loadings2d = run.pca.loadings.map((row) => [row[0], row[1] || 0]);
    const maxLoadingX = Math.max(1e-8, Math.max(...loadings2d.map((value) => Math.abs(value[0]))));
    const maxLoadingY = Math.max(1e-8, Math.max(...loadings2d.map((value) => Math.abs(value[1]))));
    const arrowScale = 0.85 * Math.min(maxScoreX / maxLoadingX, maxScoreY / maxLoadingY);

    const arrowAnnotations = loadings2d.map((loading) => ({
      x: loading[0] * arrowScale,
      y: loading[1] * arrowScale,
      ax: 0,
      ay: 0,
      xref: 'x',
      yref: 'y',
      axref: 'x',
      ayref: 'y',
      showarrow: true,
      arrowhead: 3,
      arrowsize: 1.15,
      arrowwidth: 2.2,
      arrowcolor: vectorColor,
      text: ''
    }));

    const labelCoords = loadings2d.map((loading, index) => {
      const x = loading[0] * arrowScale;
      const y = loading[1] * arrowScale;
      const length = Math.hypot(x, y);
      if (length < 1e-6) {
        const angle = (-Math.PI / 2) + (index * 2 * Math.PI) / Math.max(names.length, 1);
        return {
          x: 0.28 * Math.cos(angle),
          y: 0.28 * Math.sin(angle)
        };
      }
      return {
        x: x + (x / length) * 0.22,
        y: y + (y / length) * 0.22
      };
    });

    const traces = [
      {
        type: 'scatter',
        mode: 'markers',
        x: xScores,
        y: yScores,
        name: 'Scores',
        marker: {
          size: scoreSample.length <= 300 ? 8 : scoreSample.length <= 1000 ? 5 : 4,
          opacity: 0.58,
          color: pointColor
        },
        hovertemplate: 'PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
      },
      {
        type: 'scatter',
        mode: 'text',
        x: labelCoords.map((entry) => entry.x),
        y: labelCoords.map((entry) => entry.y),
        text: names,
        textposition: 'middle center',
        textfont: {
          color: vectorColor,
          size: 13
        },
        hoverinfo: 'skip',
        showlegend: false
      }
    ];

    const layout = {
      margin: { l: 55, r: 25, t: 20, b: 55 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { family: 'system-ui, sans-serif', size: 12, color: '#111827' },
      xaxis: {
        title: `PC1 (${formatNumber(run.pca.explained[0] * 100, 1)}%)`,
        zeroline: true
      },
      yaxis: {
        title: `PC2 (${formatNumber((run.pca.explained[1] || 0) * 100, 1)}%)`,
        zeroline: true
      },
      annotations: arrowAnnotations,
      showlegend: false
    };

    Plotly.newPlot('biplot', traces, layout, { responsive: true, displayModeBar: false });
  }

  function computeObservedMatrices(data) {
    return {
      covariance: PCAPlayground.covarianceMatrix(PCAPlayground.centerScale(data, false).data),
      correlation: PCAPlayground.correlationMatrix(data)
    };
  }

  function runAnalysis() {
    syncControlState();
    try {
      let sigmaUsed = PCAPlayground.cloneMatrix(state.sigma);
      let sigmaRepair = null;
      const validation = PCAPlayground.validateSigma(state.sigma);
      if (!validation.ok) {
        if (validation.code === 'not-psd') {
          sigmaRepair = PCAPlayground.repairSigma(state.sigma);
          if (!sigmaRepair.ok) {
            throw new Error('Sigma is not positive semidefinite and could not be repaired.');
          }
          sigmaUsed = sigmaRepair.sigma;
        } else {
          throw new Error(validation.message);
        }
      }

      const data = PCAPlayground.generateMVN({
        n: state.n,
        means: state.means,
        sigma: sigmaUsed,
        seed: state.seed
      });
      const pca = PCAPlayground.pca(data, state.scale);
      const parallel = PCAPlayground.parallelAnalysis(data, state.scale, state.seed, 100);
      const observed = computeObservedMatrices(data);
      const displayIndices = PCAPlayground.sampleIndices(data.length, 2000, state.seed);
      const displayData = displayIndices.map((index) => data[index]);
      const displayScores = displayIndices.map((index) => pca.scores[index]);
      const kaiserCount = state.scale ? pca.eigenvalues.filter((value) => value > 1).length : null;
      const parallelCount = pca.eigenvalues.filter((value, index) => value > parallel.q95[index]).length;

      state.lastRun = {
        data,
        pca,
        parallel: {
          ...parallel,
          count: parallelCount
        },
        sigmaUsed,
        sigmaRepair,
        sampleCovariance: observed.covariance,
        sampleCorrelation: observed.correlation,
        displayData,
        displayScores,
        kaiserCount
      };

      renderSummaries(state.lastRun);
      renderPreviewTables(state.lastRun);
      buildPrintout(state.lastRun);
      renderScatterMatrix(state.lastRun);
      renderScreePlot(state.lastRun);
      renderBiplot(state.lastRun);

      if (displayData.length < data.length) {
        elements.plotNote.textContent = `${displayData.length.toLocaleString()} of ${data.length.toLocaleString()} points shown in the plots.`;
        elements.plotNote.classList.remove('hidden-note');
      } else {
        elements.plotNote.textContent = '';
        elements.plotNote.classList.add('hidden-note');
      }

      if (sigmaRepair) {
        const minEigen = formatNumber(sigmaRepair.minOriginalEigenvalue, 4);
        setStatus(`Sigma was repaired for this run. Min input eigenvalue: ${minEigen}.`, 'warning');
      } else {
        setStatus();
      }
    } catch (error) {
      setStatus(error.message, 'error');
      elements.plotNote.textContent = '';
      elements.plotNote.classList.add('hidden-note');
      elements.summaryParallel.textContent = '-';
      elements.summaryKaiser.textContent = '-';
      elements.summaryPC1.textContent = '-';
      elements.summaryPC12.textContent = '-';
      elements.pcaPrintout.textContent = '';
      elements.dataPreview.innerHTML = '';
      elements.targetMatrix.innerHTML = '';
      elements.observedCovariance.innerHTML = '';
      elements.observedCorrelation.innerHTML = '';
      elements.loadingsTable.innerHTML = '';
      elements.scoresPreview.innerHTML = '';
      Plotly.purge('scatter-matrix');
      Plotly.purge('scree-plot');
      Plotly.purge('biplot');
    }
  }

  function openAboutModal() {
    elements.aboutModal.hidden = false;
    document.body.style.overflow = 'hidden';
  }

  function closeAboutModal() {
    elements.aboutModal.hidden = true;
    document.body.style.overflow = '';
  }

  function initializeControls() {
    state.p = Number(elements.variableCount.value);
    state.n = Number(elements.nSlider.value);
    state.scale = elements.scaleToggle.checked;
    state.seed = Number(elements.seedInput.value);
    state.preset = elements.presetSelect.value;
    applyPreset();
  }

  elements.variableCount.addEventListener('change', () => {
    syncControlState();
    applyPreset();
  });

  elements.presetSelect.addEventListener('change', () => {
    syncControlState();
    applyPreset();
  });

  elements.nSlider.addEventListener('input', () => {
    syncControlState();
  });

  elements.scaleToggle.addEventListener('change', syncControlState);
  elements.seedInput.addEventListener('input', syncControlState);

  document.querySelectorAll('.chip-button').forEach((button) => {
    button.addEventListener('click', () => {
      elements.nSlider.value = button.dataset.n;
      syncControlState();
    });
  });

  elements.resetButton.addEventListener('click', () => {
    syncControlState();
    applyPreset();
  });

  elements.aboutButton.addEventListener('click', openAboutModal);
  elements.closeAbout.addEventListener('click', closeAboutModal);
  elements.aboutModal.addEventListener('click', (event) => {
    if (event.target === elements.aboutModal) {
      closeAboutModal();
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !elements.aboutModal.hidden) {
      closeAboutModal();
    }
  });

  elements.runButton.addEventListener('click', async () => {
    elements.runButton.disabled = true;
    elements.runButton.textContent = 'Running...';
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    runAnalysis();
    elements.runButton.disabled = false;
    elements.runButton.textContent = 'Run PCA';
  });

  initializeControls();
  runAnalysis();
})();