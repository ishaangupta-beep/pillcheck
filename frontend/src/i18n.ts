// Patient-facing text in English and Hindi. Caregiver screens are English.
export type Lang = "en" | "hi";

const T = {
  en: {
    morning: "Good morning", afternoon: "Good afternoon", evening: "Good evening",
    nextDose: "Next dose", now: "Now", laterToday: "Later today", nextUp: "Coming up",
    tablets: (n: number) => (n === 1 ? "1 tablet" : `${n} tablets`),
    photograph: "Photograph the strip",
    takeFirst: "Take your tablet first, then the photo.",
    today: "Today", ofTaken: (a: number, b: number) => `${a} of ${b} taken`,
    allDone: "All done for today", allDoneSub: "Every dose today is recorded. Well done.",
    reassure: (n: string) => `${n} is told only if something needs attention.`,
    status: { taken: "Taken", late: "Taken late", partial: "Partly taken", missed: "Missed", due: "Due", later: "Later" } as Record<string, string>,
    capture: "Photograph the strip", fitStrip: "Fit the whole strip inside the frame",
    tips: ["From above", "No flash", "Plain surface"], gallery: "Gallery", sample: "Sample",
    checking: "Checking the strip…", which: "Which medicine?",
    samplesTitle: "Sample photos", samplesHint: "No strip at hand? Use one of these.",
    recorded: "Dose recorded", recordedLate: "Recorded (taken late)", extra: "More than scheduled",
    noneNew: "No new tablet taken", retake: "Please take another photo", wrongStrip: "This is a different medicine",
    notRecognised: "Strip not recognised", checkedAt: (t: string) => `Checked at ${t}`,
    legend: ["Still in strip", "Taken before", "Taken now"], done: "Done", tryAgain: "Take another photo", from: "PillCheck",
    noStrip: "No strip registered yet", noStripSub: "Ask your caregiver to set up your medicines.",
  },
  hi: {
    morning: "सुप्रभात", afternoon: "नमस्ते", evening: "शुभ संध्या",
    nextDose: "अगली खुराक", now: "अभी", laterToday: "आज बाद में", nextUp: "आगे",
    tablets: (n: number) => `${n} गोली`,
    photograph: "पत्ते की फ़ोटो लें",
    takeFirst: "पहले गोली लें, फिर फ़ोटो।",
    today: "आज", ofTaken: (a: number, b: number) => `${b} में से ${a} ली गईं`,
    allDone: "आज की सभी गोलियाँ हो गईं", allDoneSub: "आज की हर खुराक दर्ज है। बहुत बढ़िया।",
    reassure: (n: string) => `${n} को सिर्फ़ ज़रूरत होने पर बताया जाता है।`,
    status: { taken: "ली गई", late: "देर से ली", partial: "आधी ली", missed: "छूट गई", due: "अभी", later: "बाद में" } as Record<string, string>,
    capture: "पत्ते की फ़ोटो लें", fitStrip: "पूरा पत्ता फ़्रेम के अंदर रखें",
    tips: ["ऊपर से", "फ़्लैश नहीं", "सादी सतह"], gallery: "गैलरी", sample: "नमूना",
    checking: "पत्ता जाँचा जा रहा है…", which: "कौन-सी दवा?",
    samplesTitle: "नमूना फ़ोटो", samplesHint: "पत्ता पास नहीं है? इनमें से एक चुनें।",
    recorded: "खुराक दर्ज हो गई", recordedLate: "दर्ज (देर से ली)", extra: "निर्धारित से ज़्यादा",
    noneNew: "कोई नई गोली नहीं निकली", retake: "कृपया एक और फ़ोटो लें", wrongStrip: "यह दूसरी दवा है",
    notRecognised: "पत्ता पहचान में नहीं आया", checkedAt: (t: string) => `${t} बजे जाँचा गया`,
    legend: ["पत्ते में बाकी", "पहले ली गई", "अभी ली गई"], done: "ठीक है", tryAgain: "एक और फ़ोटो लें", from: "PillCheck",
    noStrip: "अभी कोई पत्ता दर्ज नहीं", noStripSub: "अपने परिजन से दवाइयाँ सेट करने को कहें।",
  },
};

export type Strings = typeof T.en;
export const strings = (lang: Lang): Strings => T[lang] as Strings;

export function fmtTime(hhmm: string): { time: string; ampm: string } {
  const [h, m] = hhmm.split(":").map(Number);
  return { time: `${h % 12 || 12}:${String(m).padStart(2, "0")}`, ampm: h < 12 ? "AM" : "PM" };
}

export function clockOf(iso: string, tz = "Asia/Kolkata"): string {
  return new Date(iso).toLocaleTimeString("en-IN", { hour: "numeric", minute: "2-digit", hour12: true, timeZone: tz }).replace(" am", " AM").replace(" pm", " PM");
}
