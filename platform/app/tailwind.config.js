/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        dark: { 900: "#0a0a0f", 800: "#12121a", 700: "#1a1a25" },
        accent: { DEFAULT: "#00d4aa", dark: "#00b894" },
      },
    },
  },
  plugins: [],
};
