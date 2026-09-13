/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // SatQuery AI brand palette
        brand: {
          50:  '#eef7ff',
          100: '#d8ecff',
          200: '#b9deff',
          300: '#89caff',
          400: '#52abff',
          500: '#2a87ff',
          600: '#1366f5',
          700: '#0c50e1',
          800: '#1041b6',
          900: '#133a8f',
          950: '#112557',
        },
        surface: {
          50:  '#f7f8fa',
          100: '#edf0f5',
          200: '#d5dbe6',
          300: '#b0bbcd',
          400: '#8495af',
          500: '#657896',
          600: '#50607c',
          700: '#424e65',
          800: '#394356',
          900: '#333b49',
          950: '#0f1117',
        },
        accent: {
          emerald: '#10b981',
          amber: '#f59e0b',
          rose: '#f43f5e',
          violet: '#8b5cf6',
          cyan: '#06b6d4',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'pulse-soft': 'pulseSoft 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        'glass': '0 4px 30px rgba(0, 0, 0, 0.1)',
        'glow': '0 0 20px rgba(42, 135, 255, 0.3)',
        'glow-lg': '0 0 40px rgba(42, 135, 255, 0.2)',
      },
    },
  },
  plugins: [],
}
