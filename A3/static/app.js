"use strict";
const app = document.getElementById("app");
const params = new URLSearchParams(location.search);
const AI_NOTE = "The AI can be wrong and sometimes describes things that aren't in the image. Check its evidence.";
const QUESTION = "Which country was this photo taken in?";
const PIN = `<svg width="18" height="24" viewBox="0 0 18 24" aria-hidden="true"><path fill="#f2644f" d="M9 0a9 9 0 0 0-9 9c0 6.4 9 15 9 15s9-8.6 9-15a9 9 0 0 0-9-9z"/><circle cx="9" cy="9" r="3.4" fill="#fff"/></svg>`;
let session = null;  // the server's view of this participant; every POST returns a fresh copy
let routePosition = 0;  // the pin's last position, so it can travel to the next stop

function h(tag, props = {}, ...kids) {
  const el = Object.assign(document.createElement(tag), props);
  el.append(...kids.flat().filter(k => k != null));
  return el;
}

async function api(path, body) {
  const options = body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {};
  const r = await fetch(path, options);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

const sessionPath = suffix => `/api/sessions/${session.participant_id}${suffix}`;
const conditionParam = aiAssist => `?ai-assist=${aiAssist ? "True" : "False"}`;

function page(...kids) {
  document.onkeydown = null;
  app.replaceChildren(h("main", {className: "page"}, h("div", {className: "sheet"}, ...kids)));
}

// One stop per photo. The pin stands at the stop for `position`, a 0-based photo index.
function route(position, total) {
  const percent = p => `${(p / (total - 1)) * 100}%`;
  const walked = h("div", {className: "route-walked"});
  const pin = h("div", {className: "route-pin", innerHTML: PIN});
  walked.style.width = pin.style.left = percent(routePosition);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    walked.style.width = pin.style.left = percent(position);
  }));
  routePosition = position;
  const stops = Array.from({length: total}, (_, i) => h("i", {className: i < position ? "walked" : ""}));
  return h("div", {className: "route", role: "img", ariaLabel: `Photo ${position + 1} of ${total}`},
           h("div", {className: "route-line"}), walked, h("div", {className: "route-stops"}, stops), pin);
}

function aiAssistFromUrl() {
  const value = (params.get("ai-assist") || "").toLowerCase();
  return value === "true" ? true : value === "false" ? false : null;
}

function invalidLink() {
  page(h("h1", {}, "This link is incomplete"),
       h("p", {}, "Open the full study link you were given."));
}

function consent(aiAssist) {
  const UNIQNAME_HELP = "Enter your uniqname: 3 to 8 letters, the part before @umich.edu.";
  const err = h("p", {className: "err", role: "alert"});
  const input = h("input", {id: "uniqname", name: "uniqname", autocomplete: "username", autocapitalize: "none",
                            spellcheck: false, required: true, placeholder: "e.g. jsmith"});
  const start = h("button", {className: "primary", type: "submit", textContent: "I agree"});
  page(
    h("h1", {}, "Where was this photo taken?"),
    h("p", {className: "lead"}, "A short study for a university course. You'll look at street-level photos and guess the country."),
    h("ul", {className: "facts"}, ["Under 10 minutes", "Stop any time"].map(fact => h("li", {}, fact))),
    h("p", {className: "muted"}, "Participation is voluntary. We record your uniqname, answers, response times and " +
      "survey ratings. Your uniqname is used only to match your two sessions. To stop, close this tab. By continuing, " +
      "you confirm that you are 18 or older."),
    h("form", {className: "field", async onsubmit(e) {
      e.preventDefault();
      const uniqname = input.value.trim().toLowerCase().replace(/@umich\.edu$/, "");
      if (!/^[a-z]{3,8}$/.test(uniqname)) {
        err.textContent = UNIQNAME_HELP;
        return input.focus();
      }
      start.disabled = true;
      try {
        session = await api("/api/sessions", {ai_assist: aiAssist, uniqname});
        // a refresh resumes this session; entering the same uniqname on this link does too
        history.replaceState(null, "", `${conditionParam(aiAssist)}&pid=${session.participant_id}`);
        session.done ? next() : instructions();
      } catch {
        err.textContent = "Could not start. Check your connection and try again.";
        start.disabled = false;
      }
    }},
      h("label", {htmlFor: "uniqname"}, "Your uniqname"),
      h("div", {className: "uniq-row"}, input, h("span", {ariaHidden: "true"}, "@umich.edu")),
      err, start));
  input.focus({preventScroll: true});
}

function instructions() {
  const total = session.images.length;
  page(
    h("h1", {}, "How it works"),
    route(0, total),
    h("ul", {className: "points"},
      h("li", {}, h("strong", {}, `Guess the country of ${total} photos`),
        `Each was taken in one of these: ${session.countries.join(", ")}. You won't be told whether you were right.`),
      session.ai_assist
        ? h("li", {}, h("strong", {}, "An AI suggests three countries"),
            "It shows its evidence, but it can be wrong. The final choice is yours.")
        : null,
      h("li", {}, h("strong", {}, "Look closely"),
        "Scroll or pinch to zoom into a photo, and drag to move around."),
      h("li", {}, h("strong", {}, "Use your own judgment"),
        "Please don't look anything up. Two simple attention checks appear along the way. Missing both ends the study, and you would need to start over.")),
    h("button", {className: "primary", textContent: "Start", onclick: next}));
}

function next() {
  if (session.terminated) return terminated();
  if (session.attention_check) return attentionCheck(session.attention_check);
  if (session.done < session.images.length) return trial(session.done);
  return session.survey_done ? thanks() : survey();
}

function attentionCheck(check) {
  const screen = photoScreen({
    src: check.image,
    label: "Quick check",
    position: session.done,
    preload: session.images[session.done],
    async onNext(answer) {
      const {passed, ...updated} = await api(sessionPath("/attention"), {check_index: check.index, answer});
      session = updated;
      passed === false && !session.terminated ? attentionWarning() : next();
    },
  });
  screen.photoLoaded.then(() => {
    screen.choices.classList.add("unaided");
    screen.choices.replaceChildren(h("p", {className: "q"}, check.question),
                                   ...check.options.map((c, i) => screen.option(c, i + 1)));
  }, () => { screen.err.textContent = "Could not load the photo. Reload the page."; });
}

function attentionWarning() {
  page(h("h1", {}, "That wasn't the expected answer"),
       h("p", {}, "Reminder: for each photo, pick the country where it was taken. Read each question carefully."),
       h("p", {}, "If you miss the other check too, the study ends and you would need to start over."),
       h("button", {className: "primary", textContent: "Continue", onclick: next}));
}

function terminated() {
  page(h("h1", {}, "This session has ended"),
       h("p", {}, "You missed both attention checks, so this session won't be used. To take part, start over."),
       h("button", {className: "primary", textContent: "Start over", onclick() {
         location.search = conditionParam(session.ai_assist);
       }}));
}

// The screen shared by trials and attention checks: photo with zoom, answer panel, Next, and keyboard shortcuts.
function photoScreen({src, label, position, preload, onNext}) {
  const total = session.images.length;
  let selected = null, busy = false;

  const img = h("img", {className: "photo", alt: "Street-level photo to locate", draggable: false});
  const level = h("span", {className: "zoom-level", ariaLive: "polite", textContent: "100%"});
  const zoomOutBtn = h("button", {className: "zoom-step", textContent: "−", ariaLabel: "Zoom out", disabled: true});
  const zoomInBtn = h("button", {className: "zoom-step", textContent: "+", ariaLabel: "Zoom in"});
  const fitBtn = h("button", {textContent: "Fit", ariaLabel: "Fit photo to frame", disabled: true});
  const stage = h("div", {className: "stage"}, img,
    h("div", {className: "zoom-hint"}, "Scroll to zoom, drag to move"),
    h("div", {className: "zoom-controls"}, zoomOutBtn, level, zoomInBtn, fitBtn));
  const zoom = attachZoom(stage, img, scale => {
    level.textContent = `${Math.round(scale * 100)}%`;
    stage.classList.toggle("is-zoomed", scale > 1);
    zoomOutBtn.disabled = fitBtn.disabled = scale <= 1;
    zoomInBtn.disabled = scale >= 5;
  });
  zoomInBtn.onclick = zoom.zoomIn;
  zoomOutBtn.onclick = zoom.zoomOut;
  fitBtn.onclick = zoom.reset;

  const photoLoaded = new Promise((resolve, reject) => {
    img.onload = () => {
      stage.style.setProperty("--ratio", img.naturalWidth / img.naturalHeight);  // the frame takes each photo's shape
      stage.classList.add("is-loaded");
      resolve();
    };
    img.onerror = reject;
  });
  img.src = src;

  const choices = h("div", {className: "rail-body"});
  const err = h("p", {className: "err", role: "alert"});
  const nextBtn = h("button", {className: "primary", textContent: "Next", disabled: true, onclick: submit});

  function option(country, key, ...kids) {
    return h("button", {className: "opt", ariaPressed: "false", onclick() {
      selected = country;
      choices.querySelectorAll(".opt").forEach(b => { b.ariaPressed = String(b === this); });
      nextBtn.disabled = false;
    }}, h("span", {className: "opt-row"},
          h("span", {className: "opt-key", ariaHidden: "true"}, String(key)),
          h("span", {className: "opt-country"}, country),
          kids[0] ?? null),
      kids.slice(1));
  }

  function clearSelection() {
    selected = null;
    nextBtn.disabled = true;
  }

  async function submit() {
    if (!selected || busy) return;
    busy = nextBtn.disabled = true;
    try {
      await onNext(selected);
    } catch {
      err.textContent = "Could not save your answer. Check your connection and press Next again.";
      busy = nextBtn.disabled = false;
    }
  }

  app.replaceChildren(h("main", {className: "trial"},
    h("header", {className: "topbar"},
      h("div", {className: "count"}, label),
      route(position, total)),
    h("div", {className: "workspace"},
      h("div", {className: "stage-frame"}, stage),
      h("section", {className: "rail", ariaLabel: "Your answer"},
        choices,
        h("div", {className: "rail-foot"}, err, nextBtn,
          h("div", {className: "keys"}, "Press a number to choose, Enter to continue"))))));
  document.onkeydown = e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "+" || e.key === "=") zoom.zoomIn();
    else if (e.key === "-") zoom.zoomOut();
    else if (e.key === "0") zoom.reset();
    else if (/^[1-9]$/.test(e.key)) {
      document.activeElement.blur();
      choices.querySelectorAll(".opt")[+e.key - 1]?.click();
    }
    // Enter submits, except on a non-choice button or an unselected choice (those handle Enter natively)
    else if (e.key === "Enter" && !e.target.closest?.("button:not(.opt), .opt[aria-pressed=false]")) {
      e.preventDefault();
      submit();
    }
  };
  if (preload) new Image().src = preload;
  return {photoLoaded, choices, err, option, clearSelection};
}

function trial(index) {
  let startedAt = performance.now(), expandedOther = false;
  const screen = photoScreen({
    src: session.images[index],
    label: [`Photo ${index + 1} `, h("span", {}, `of ${session.images.length}`)],
    position: index,
    preload: session.images[index + 1],
    async onNext(response) {
      const rt_ms = Math.min(3600000, Math.round(performance.now() - startedAt));
      session = await api(sessionPath("/responses"), {trial_index: index, response, expanded_other: expandedOther, rt_ms});
      next();
    },
  });
  const {choices, err, option, photoLoaded} = screen;

  async function loadUnaided() {
    try {
      await photoLoaded;
      startedAt = performance.now();  // the response time starts once the photo and choices are on screen
      choices.classList.add("unaided");
      choices.replaceChildren(h("p", {className: "q"}, QUESTION), ...session.countries.map((c, i) => option(c, i + 1)));
    } catch {
      err.textContent = "Could not load the photo. Reload the page.";
    }
  }

  function renderCards(cards) {
    const suggested = cards.map(c => c.country);
    const rest = session.countries.filter(c => !suggested.includes(c));
    choices.replaceChildren(
      h("p", {className: "q"}, QUESTION),
      h("p", {className: "ai-note"}, AI_NOTE),
      ...cards.map((c, i) => {
        const percent = Math.round(c.p * 100);
        const fill = h("i");
        fill.style.width = `${Math.max(1, percent)}%`;
        return option(c.country, i + 1,
          h("span", {className: "opt-percent"}, `${percent}%`),
          h("span", {className: "meter"}, fill),
          h("span", {className: "why"}, c.rationale));
      }),
      ...(expandedOther ? [h("div", {className: "other-options"}, rest.map((c, i) => option(c, cards.length + i + 1)))]
                        : [h("button", {className: "text-button", textContent: "None of these — choose another", onclick() {
                            expandedOther = true;
                            screen.clearSelection();
                            renderCards(cards);
                          }})]));
  }

  async function loadCards() {
    err.replaceChildren();
    choices.replaceChildren(h("p", {className: "q"}, QUESTION),
                            h("p", {className: "thinking", role: "status"}, "The AI is looking at the photo…"),
                            h("div", {className: "ghost"}), h("div", {className: "ghost"}), h("div", {className: "ghost"}));
    try {
      const [cards] = await Promise.all([api(sessionPath(`/ai/${index}`)), photoLoaded]);
      startedAt = performance.now();  // the response time starts once the photo and suggestions are on screen
      renderCards(cards);
    } catch {
      choices.replaceChildren(h("p", {className: "q"}, QUESTION));
      err.replaceChildren("Could not load the AI's suggestions. ",
                          h("button", {className: "text-button", textContent: "Try again", onclick: loadCards}));
    }
  }

  session.ai_assist ? loadCards() : loadUnaided();
}

function survey() {
  const answers = {};
  const required = session.ai_assist ? 6 : 4;
  const finish = h("button", {className: "primary", textContent: "Finish", disabled: true});
  const text = h("textarea", {rows: 3, maxLength: 5000, id: "free"});
  const err = h("p", {className: "err", role: "alert"});
  const answer = (key, value, button, buttons) => {
    answers[key] = value;
    buttons.forEach(b => { b.ariaPressed = String(b === button); });
    finish.disabled = Object.keys(answers).length < required;
  };
  const scale = (key, label, low, high) => {
    const buttons = [1, 2, 3, 4, 5, 6, 7].map(v => h("button", {textContent: v, ariaPressed: "false", ariaLabel: `${v} of 7`,
      onclick() { answer(key, v, this, buttons); }}));
    return h("fieldset", {}, h("legend", {}, label), h("div", {className: "segs"}, buttons),
             h("div", {className: "ends", ariaHidden: "true"}, h("span", {}, low), h("span", {}, high)));
  };
  const yesNo = (key, label) => {
    const buttons = [["Yes", true], ["No", false]].map(([name, v]) => h("button", {textContent: name, ariaPressed: "false",
      onclick() { answer(key, v, this, buttons); }}));
    return h("fieldset", {}, h("legend", {}, label), h("div", {className: "segs two"}, buttons));
  };
  finish.onclick = async () => {
    finish.disabled = true;
    try {
      session = await api(sessionPath("/survey"), {...answers, free_text: text.value});
      thanks();
    } catch {
      err.textContent = "Could not save. Check your connection and try again.";
      finish.disabled = false;
    }
  };
  page(h("h1", {}, "Almost done"),
    scale("confidence", "How confident are you in your answers overall?", "1 · Not at all", "7 · Very confident"),
    scale("mental_demand", "How mentally demanding was this task?", "1 · Very low", "7 · Very high"),
    session.ai_assist ? [
      scale("trust", "How much did you trust the AI's suggestions?", "1 · Not at all", "7 · Completely"),
      scale("usefulness", "How useful were the AI's suggestions?", "1 · Not at all", "7 · Very useful"),
    ] : null,
    scale("geo_knowledge", "How would you rate your own geography knowledge?", "1 · Very low", "7 · Expert"),
    yesNo("seen_other_condition", `Have you already done the version of this study ${session.ai_assist ? "without" : "with"} AI suggestions?`),
    h("div", {className: "free"},
      h("label", {htmlFor: "free"}, "Anything else to tell us? ", h("small", {}, "Optional. Include anything confusing or broken.")), text),
    err, finish);
}

function thanks() {
  const total = session.images.length;
  routePosition = total - 1;
  page(h("h1", {}, "Thank you!"),
       route(total - 1, total),
       h("p", {}, "Your answers are saved. You can close this tab now."),
       h("p", {}, "If a spreadsheet brought you here, add your name and comments to it."),
       h("p", {className: "muted"}, "Recorded as"),
       h("p", {className: "code"}, session.participant_id));
}

(async () => {
  const pid = params.get("pid");
  if (pid) {
    session = await api(`/api/sessions/${encodeURIComponent(pid)}`).catch(() => null);
    if (session) return session.done ? next() : instructions();
  }
  const aiAssist = aiAssistFromUrl();
  aiAssist === null ? invalidLink() : consent(aiAssist);
})();
