/* LioScan — card detection worker (OpenCV.js).
 * detect: finds the outline of a Pokémon card (4 corners) in a small video frame.
 *         Every rectangle is checked against the layout of a Pokémon card
 *         (uniform border, text band at the top and at the bottom, picture in the middle)
 *         so that shelves, posters, windows or screens are ignored.
 * warp:   flattens the card from a full-resolution frame to a 630×880 image.
 */
/* global cv */
let ready = null

function init() {
  if (ready) return ready
  ready = new Promise((resolve, reject) => {
    try {
      self.importScripts('/vendor/opencv.js')
    } catch (e) {
      reject(e)
      return
    }
    const done = () => resolve()
    const c = self.cv
    if (c && typeof c.then === 'function') {
      c.then((m) => {
        self.cv = m
        done()
      })
    } else if (c && c.Mat) done()
    else if (c) c.onRuntimeInitialized = done
    else reject(new Error('OpenCV indisponible'))
  })
  return ready
}

const CARD = 63 / 88
const VW = 200 // verification warp size
const VH = 280
const MIN_SCORE = 0.65

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1])
const clamp01 = (v) => Math.max(0, Math.min(1, v))

/** Order corners: top-left, top-right, bottom-right, bottom-left. */
function order(pts) {
  const bySum = [...pts].sort((a, b) => a[0] + a[1] - (b[0] + b[1]))
  const byDiff = [...pts].sort((a, b) => a[1] - a[0] - (b[1] - b[0]))
  return [bySum[0], byDiff[0], bySum[3], byDiff[3]]
}

function sides(q) {
  return { w: (dist(q[0], q[1]) + dist(q[3], q[2])) / 2, h: (dist(q[0], q[3]) + dist(q[1], q[2])) / 2 }
}

function ratioOk(q) {
  const { w, h } = sides(q)
  if (!w || !h) return false
  return Math.abs(Math.min(w, h) / Math.max(w, h) - CARD) < 0.1
}

/** Portrait orientation: the long side becomes vertical. */
const portrait = (q) => (sides(q).w > sides(q).h ? [q[3], q[0], q[1], q[2]] : q)

const shoelace = (q) => Math.abs(q.reduce((s, p, i) => s + p[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * p[1], 0)) / 2

function matPoints(m) {
  const out = []
  for (let i = 0; i < m.rows; i++) out.push([m.data32S[i * 2], m.data32S[i * 2 + 1]])
  return out
}

function median(mat) {
  const d = mat.data
  const hist = new Uint32Array(256)
  let n = 0
  for (let i = 0; i < d.length; i += 7) {
    hist[d[i]]++
    n++
  }
  let acc = 0
  for (let v = 0; v < 256; v++) {
    acc += hist[v]
    if (acc >= n / 2) return v
  }
  return 128
}

/* ---------------- candidate rectangles ---------------- */

function contourQuads(edges, frame, out) {
  const contours = new cv.MatVector()
  const hier = new cv.Mat()
  try {
    cv.findContours(edges, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i)
      const area = cv.contourArea(c)
      if (area < frame * 0.03 || area > frame * 0.9) {
        c.delete()
        continue
      }
      const peri = cv.arcLength(c, true)
      let quad = null
      for (const eps of [0.02, 0.035]) {
        const approx = new cv.Mat()
        cv.approxPolyDP(c, approx, eps * peri, true)
        if (approx.rows === 4 && cv.isContourConvex(approx)) quad = order(matPoints(approx))
        approx.delete()
        if (quad) break
      }
      if (!quad) {
        const rr = cv.minAreaRect(c)
        const boxArea = rr.size.width * rr.size.height
        if (boxArea > 0 && area / boxArea > 0.82) quad = order(cv.RotatedRect.points(rr).map((pt) => [pt.x, pt.y]))
      }
      if (quad && ratioOk(quad)) out.push({ quad, area: shoelace(quad) })
      c.delete()
    }
  } finally {
    contours.delete()
    hier.delete()
  }
}

function intersect(a, b) {
  const [x1, y1, x2, y2] = a
  const [x3, y3, x4, y4] = b
  const d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
  if (Math.abs(d) < 1e-6) return null
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
  return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
}

/** Rebuild rectangles from long straight edges (outline partly hidden by fingers). */
function lineQuads(edges, w, h, out) {
  const lines = new cv.Mat()
  cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 30, Math.min(w, h) * 0.18, 12)
  const segs = []
  for (let i = 0; i < lines.rows; i++) {
    const l = [lines.data32S[i * 4], lines.data32S[i * 4 + 1], lines.data32S[i * 4 + 2], lines.data32S[i * 4 + 3]]
    let ang = Math.atan2(l[3] - l[1], l[2] - l[0])
    if (ang < 0) ang += Math.PI
    segs.push({ l, len: Math.hypot(l[2] - l[0], l[3] - l[1]), ang })
  }
  lines.delete()
  if (segs.length < 4) return
  const bins = new Float64Array(90)
  for (const s of segs) bins[Math.floor(((s.ang * 180) / Math.PI) % 90)] += s.len
  let dom = 0
  const win = (i) => bins[i] + bins[(i + 1) % 90] + bins[(i + 89) % 90]
  for (let i = 0; i < 90; i++) if (win(i) > win(dom)) dom = i
  const near = (a, target) => {
    let d = Math.abs(((a * 180) / Math.PI - target) % 180)
    if (d > 90) d = 180 - d
    return d < 10
  }
  const pos = (s, t) => ((s.l[0] + s.l[2]) / 2) * -Math.sin(t) + ((s.l[1] + s.l[3]) / 2) * Math.cos(t)
  const pairs = (group, t) => {
    const sorted = group.map((s) => ({ s, p: pos(s, t) })).sort((a, b) => a.p - b.p)
    const res = []
    for (let i = 0; i < sorted.length; i++)
      for (let j = i + 1; j < sorted.length; j++) {
        const gap = sorted[j].p - sorted[i].p
        if (gap > Math.min(w, h) * 0.15) res.push({ a: sorted[i].s, b: sorted[j].s, gap, support: sorted[i].s.len + sorted[j].s.len })
      }
    return res.sort((x, y) => y.support - x.support).slice(0, 12)
  }
  const tA = (dom * Math.PI) / 180
  const pa = pairs(segs.filter((s) => near(s.ang, dom)), tA)
  const pb = pairs(segs.filter((s) => near(s.ang, dom + 90)), tA + Math.PI / 2)
  for (const A of pa)
    for (const B of pb) {
      const r = Math.min(A.gap, B.gap) / Math.max(A.gap, B.gap)
      if (Math.abs(r - CARD) > 0.09) continue
      const c = [intersect(A.a.l, B.a.l), intersect(A.a.l, B.b.l), intersect(A.b.l, B.b.l), intersect(A.b.l, B.a.l)]
      if (c.some((p) => !p || p[0] < -w * 0.05 || p[1] < -h * 0.05 || p[0] > w * 1.05 || p[1] > h * 1.05)) continue
      const quad = order(c)
      // every edge must really be visible on a good part of its length
      // (lines of pair A run along a side whose length is B's spacing, and vice versa)
      if (Math.min(A.a.len, A.b.len) < B.gap * 0.3 || Math.min(B.a.len, B.b.len) < A.gap * 0.3) continue
      if (ratioOk(quad)) out.push({ quad, area: shoelace(quad), lines: true })
    }
}

/* ---------------- "is this a Pokémon card?" ---------------- */

/**
 * Score 0..1 from the layout every Pokémon card shares:
 *  - a thin, uniform border all around (yellow on older cards, silver/grey since Scarlet & Violet — full arts included)
 *  - text in the top band (name, "Base"/stage, HP + energy symbol) and in the bottom band (set, number, illustrator)
 *  - a busy picture in the middle
 * Text bands are measured on the full width so the score does not depend on the card being upside down.
 */
function cardScore(src, quad) {
  const q = portrait(quad)
  const from = cv.matFromArray(4, 1, cv.CV_32FC2, q.flat())
  const to = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, VW, 0, VW, VH, 0, VH])
  const M = cv.getPerspectiveTransform(from, to)
  const warped = new cv.Mat()
  let d
  try {
    cv.warpPerspective(src, warped, M, new cv.Size(VW, VH), cv.INTER_LINEAR, cv.BORDER_REPLICATE)
    d = warped.data.slice()
  } finally {
    from.delete()
    to.delete()
    M.delete()
    warped.delete()
  }
  const px = (x, y) => (y * VW + x) * 4
  const gray = (x, y) => {
    const i = px(x, y)
    return 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
  }

  // 1. border: a smooth band along each side (flat yellow on older cards, metallic silver gradient since
  //    Scarlet & Violet). "Smooth" = tiny steps between neighbouring pixels. One side may be hidden by fingers.
  const C = 14 // skip rounded corners
  const smoothSide = (pts) => {
    let ok = 0
    let n = 0
    for (let k = 1; k < pts.length; k++) {
      n++
      if (Math.abs(gray(pts[k][0], pts[k][1]) - gray(pts[k - 1][0], pts[k - 1][1])) < 7) ok++
    }
    return n ? ok / n : 0
  }
  const sideU = []
  const borderPx = []
  for (const side of [0, 1, 2, 3]) {
    let best = 0
    for (const o of [2, 3, 4]) {
      const pts = []
      if (side === 0) for (let x = C; x < VW - C; x++) pts.push([x, o])
      if (side === 2) for (let x = C; x < VW - C; x++) pts.push([x, VH - 1 - o])
      if (side === 1) for (let y = C; y < VH - C; y++) pts.push([VW - 1 - o, y])
      if (side === 3) for (let y = C; y < VH - C; y++) pts.push([o, y])
      best = Math.max(best, smoothSide(pts))
      if (o === 3) for (let k = 0; k < pts.length; k += 3) borderPx.push(px(pts[k][0], pts[k][1]))
    }
    sideU.push(best)
  }
  sideU.sort((p, q) => q - p)
  const uniform = (sideU[0] + sideU[1] + sideU[2]) / 3

  // inner frame line: where the border meets the artwork/text box there is a sharp line parallel to the edge
  const lineSide = (side) => {
    let best = 0
    for (let o = 4; o <= 16; o++) {
      let hit = 0
      let n = 0
      const len = side % 2 === 0 ? VW : VH
      for (let t = C; t < len - C; t += 2) {
        let a
        let b
        if (side === 0) [a, b] = [gray(t, o), gray(t, o + 3)]
        else if (side === 2) [a, b] = [gray(t, VH - 1 - o), gray(t, VH - 4 - o)]
        else if (side === 1) [a, b] = [gray(VW - 1 - o, t), gray(VW - 4 - o, t)]
        else [a, b] = [gray(o, t), gray(o + 3, t)]
        n++
        if (Math.abs(a - b) > 14) hit++
      }
      best = Math.max(best, hit / n)
    }
    return best
  }
  const lines4 = [0, 1, 2, 3].map(lineSide).sort((p, q) => q - p)
  const frameLine = (lines4[0] + lines4[1] + lines4[2]) / 3
  const medOf = (j) => borderPx.map((i) => d[i + j]).sort((p, q) => p - q)[borderPx.length >> 1]
  const mr = medOf(0)
  const mg = medOf(1)
  const mb = medOf(2)

  // border colour: Pokémon borders are yellow (older cards) or silver / grey / white (recent cards)
  const mx = Math.max(mr, mg, mb)
  const mn = Math.min(mr, mg, mb)
  const sat = mx ? (mx - mn) / mx : 0
  let hue = 0
  if (mx !== mn) {
    if (mx === mr) hue = (60 * ((mg - mb) / (mx - mn)) + 360) % 360
    else if (mx === mg) hue = 60 * ((mb - mr) / (mx - mn)) + 120
    else hue = 60 * ((mr - mg) / (mx - mn)) + 240
  }
  const colourOk = sat < 0.28 || (hue >= 30 && hue <= 70) ? 1 : 0

  // 2. text density in a band: share of strong horizontal contrasts
  const density = (y0, y1) => {
    let n = 0
    let hit = 0
    for (let y = Math.round(VH * y0); y < Math.round(VH * y1); y++)
      for (let x = Math.round(VW * 0.05); x < Math.round(VW * 0.95) - 2; x++) {
        n++
        if (Math.abs(gray(x + 2, y) - gray(x, y)) > 20) hit++
      }
    return hit / n
  }
  const top = density(0.03, 0.1)
  const bottom = density(0.9, 0.97)
  const textScore = (v) => (v < 0.04 ? v / 0.04 : v > 0.55 ? 0.4 : 1)

  // 3. picture in the middle
  let s = 0
  let s2 = 0
  let n = 0
  for (let y = Math.round(VH * 0.15); y < Math.round(VH * 0.55); y += 2)
    for (let x = Math.round(VW * 0.15); x < Math.round(VW * 0.85); x += 2) {
      const g = gray(x, y)
      s += g
      s2 += g * g
      n++
    }
  const sd = Math.sqrt(Math.max(0, s2 / n - (s / n) ** 2))

  const tTop = textScore(top)
  const tBottom = textScore(bottom)
  const score =
    0.2 * clamp01((uniform - 0.6) / 0.3) +
    0.25 * clamp01((frameLine - 0.4) / 0.35) +
    0.1 * colourOk +
    0.25 * Math.min(tTop, tBottom) +
    0.1 * ((tTop + tBottom) / 2) +
    0.1 * clamp01(sd / 22)
  return { score: frameLine < 0.35 ? Math.min(score, 0.4) : score, uniform, frameLine, top, bottom, sd, colourOk, sat, hue, sideU, lines4 }
}

/* ---------------- helpers ---------------- */

function insideQuad(p, q) {
  // point inside a convex quad (same side of every edge), with a small tolerance
  let sign = 0
  for (let i = 0; i < 4; i++) {
    const a = q[i]
    const b = q[(i + 1) % 4]
    const cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if (Math.abs(cross) < 1e-6) continue
    const s = Math.sign(cross)
    if (sign === 0) sign = s
    else if (s !== sign) {
      // allow points within 2 % of the edge length
      const len = Math.hypot(b[0] - a[0], b[1] - a[1])
      if (Math.abs(cross) / len > len * 0.02) return false
    }
  }
  return true
}

/** Sharpness of the card area: variance of the Laplacian (higher = sharper). */
function sharpness(gray, quad) {
  const xs = quad.map((p) => p[0])
  const ys = quad.map((p) => p[1])
  const x0 = Math.max(1, Math.round(Math.min(...xs)))
  const y0 = Math.max(1, Math.round(Math.min(...ys)))
  const x1 = Math.min(gray.cols - 2, Math.round(Math.max(...xs)))
  const y1 = Math.min(gray.rows - 2, Math.round(Math.max(...ys)))
  const d = gray.data
  const W = gray.cols
  let s = 0
  let s2 = 0
  let n = 0
  for (let y = y0; y < y1; y += 2)
    for (let x = x0; x < x1; x += 2) {
      const i = y * W + x
      const l = 4 * d[i] - d[i - 1] - d[i + 1] - d[i - W] - d[i + W]
      s += l
      s2 += l * l
      n++
    }
  return n ? s2 / n - (s / n) ** 2 : 0
}

/* ---------------- detection ---------------- */

function detect(image, roi, hi, debug) {
  const src = cv.matFromImageData(image)
  const gray = new cv.Mat()
  const edges = new cv.Mat()
  const k3 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3))
  const k7 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7))
  const W = image.width
  const H = image.height
  const frame = W * H
  const r = roi || { x: 0, y: 0, w: W, h: H }
  const hiSrc = hi ? cv.matFromImageData(hi) : null
  const k = hi ? hi.width / W : 1
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0)
    const m = median(gray)
    const cands = []
    for (const [lo, hi, kernel, close] of [
      [35, 110, k3, false],
      [Math.max(10, 0.66 * m), Math.min(255, 1.33 * m), k3, false],
      [20, 70, k7, true],
    ]) {
      cv.Canny(gray, edges, lo, hi)
      if (close) cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel)
      else cv.dilate(edges, edges, kernel)
      contourQuads(edges, frame, cands)
    }
    cv.Canny(gray, edges, 30, 100)
    lineQuads(edges, W, H, cands)

    // keep rectangles centred in the aiming frame and big enough
    const minArea = r.w * r.h * 0.12
    const inRoi = cands.filter((c) => {
      const cx = c.quad.reduce((a, p) => a + p[0], 0) / 4
      const cy = c.quad.reduce((a, p) => a + p[1], 0) / 4
      const tx = r.w * 0.08
      const ty = r.h * 0.08
      const inside = c.quad.every((p) => p[0] > r.x - tx && p[0] < r.x + r.w + tx && p[1] > r.y - ty && p[1] < r.y + r.h + ty)
      return inside && cx > r.x && cx < r.x + r.w && cy > r.y && cy < r.y + r.h && c.area >= minArea
    })
    // dedupe near-identical quads, biggest first
    inRoi.sort((a, b) => b.area - a.area)
    const uniq = []
    for (const c of inRoi) if (!uniq.some((u) => c.quad.every((p, i) => dist(p, u.quad[i]) < Math.min(W, H) * 0.03))) uniq.push(c)

    let best = null
    const valid = []
    const dbg = debug ? { total: cands.length, inRoi: inRoi.length, scored: [] } : null
    for (const c of uniq.slice(0, 10)) {
      const v = cardScore(hiSrc || src, hiSrc ? c.quad.map((p) => [p[0] * k, p[1] * k]) : c.quad)
      if (dbg) dbg.scored.push({ quad: c.quad.map((p) => p.map(Math.round)), lines: !!c.lines, area: Math.round(c.area), score: +v.score.toFixed(2), u: +v.uniform.toFixed(2), line: +v.frameLine.toFixed(2), top: +v.top.toFixed(3), bot: +v.bottom.toFixed(3), sd: Math.round(v.sd), col: v.colourOk })
      if (v.score < MIN_SCORE - 0.1) continue
      valid.push({ ...c, ...v })
      if (v.score < MIN_SCORE) continue
      const rank = v.score + (c.area / (r.w * r.h)) * 0.3
      if (!best || rank > best.rank) best = { ...c, ...v, rank }
    }
    // card in a sleeve / binder pocket: two nested card-shaped outlines → the inner one is the card
    if (best) {
      const inner = valid
        .filter((c) => c !== best && c.area < best.area * 0.97 && c.area > best.area * 0.6 && c.quad.every((p) => insideQuad(p, best.quad)))
        .sort((a, b) => b.area - a.area)[0]
      if (inner) best = { ...inner, rank: best.rank }
    }
    let sharp = 0
    if (best) {
      // measured on the detailed copy, before any blur
      const g2 = new cv.Mat()
      cv.cvtColor(hiSrc || src, g2, cv.COLOR_RGBA2GRAY)
      sharp = sharpness(g2, hiSrc ? best.quad.map((p) => [p[0] * k, p[1] * k]) : best.quad)
      g2.delete()
    }
    const result = best ? { quad: best.quad, coverage: best.area / frame, score: best.score, sharp } : null
    return dbg ? { result, dbg } : result
  } finally {
    src.delete()
    if (hiSrc) hiSrc.delete()
    gray.delete()
    edges.delete()
    k3.delete()
    k7.delete()
  }
}

/**
 * Flattens the card at the definition it really has in the source (up to 2.5 × 630×880), then
 * shrinks it to 630×880 with area averaging. Warping straight to 630×880 samples only a few source
 * pixels per output pixel (aliasing: small text turns to noise). Returns both: the full-definition
 * copy is used to read the small text (attacks, card number).
 */
function warp(image, quad, withDetail) {
  const W = 630
  const H = 880
  const q = portrait(quad)
  const side = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1])
  const tall = Math.max(side(q[0], q[3]), side(q[1], q[2]))
  const k = Math.min(2.5, Math.max(1, tall / H))
  const DW = Math.round(W * k)
  const DH = Math.round(H * k)
  const src = cv.matFromImageData(image)
  const from = cv.matFromArray(4, 1, cv.CV_32FC2, q.flat())
  const to = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, DW, 0, DW, DH, 0, DH])
  const M = cv.getPerspectiveTransform(from, to)
  const big = new cv.Mat()
  const out = new cv.Mat()
  try {
    cv.warpPerspective(src, big, M, new cv.Size(DW, DH), cv.INTER_LINEAR, cv.BORDER_REPLICATE)
    if (k > 1) cv.resize(big, out, new cv.Size(W, H), 0, 0, cv.INTER_AREA)
    else big.copyTo(out)
    return {
      image: new ImageData(new Uint8ClampedArray(out.data), W, H),
      detail: withDetail && k > 1.15 ? new ImageData(new Uint8ClampedArray(big.data), DW, DH) : null,
    }
  } finally {
    src.delete()
    from.delete()
    to.delete()
    M.delete()
    big.delete()
    out.delete()
  }
}

/** Debug helper: score a given quad (used by tests). */
function score(image, quad) {
  const src = cv.matFromImageData(image)
  try {
    return cardScore(src, quad)
  } finally {
    src.delete()
  }
}

self.onmessage = async (e) => {
  const { id, type } = e.data
  try {
    await init()
    if (type === 'init') return self.postMessage({ id, type, ok: true })
    if (type === 'detect') return self.postMessage({ id, type, result: detect(e.data.image, e.data.roi, e.data.hi, e.data.debug) })
    if (type === 'score') return self.postMessage({ id, type, result: score(e.data.image, e.data.quad) })
    if (type === 'warp') {
      const { image, detail } = warp(e.data.image, e.data.quad, e.data.detail)
      return self.postMessage({ id, type, image, detail }, detail ? [image.data.buffer, detail.data.buffer] : [image.data.buffer])
    }
  } catch (err) {
    self.postMessage({ id, type, error: String((err && err.message) || err) })
  }
}
