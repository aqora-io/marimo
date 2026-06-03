const round = (num) =>
  num
    .toFixed(7)
    .replace(/(\.[0-9]+?)0+$/, "$1")
    .replace(/\.0$/, "");
const rem = (px) => `calc(var(--unit) * ${round(px / 16)})`;

export const prose = {
  sm: {
    css: [
      {
        fontSize: rem(14),
        kbd: {
          borderRadius: rem(5),
        },
        pre: {
          borderRadius: rem(4),
        },
      },
    ],
  },
  base: {
    css: [
      {
        fontSize: rem(16),
        kbd: {
          borderRadius: rem(5),
        },
        pre: {
          borderRadius: rem(6),
        },
      },
    ],
  },
  lg: {
    css: [
      {
        fontSize: rem(18),
        kbd: {
          borderRadius: rem(5),
        },
        pre: {
          borderRadius: rem(6),
        },
      },
    ],
  },
  xl: {
    css: [
      {
        fontSize: rem(20),
        kbd: {
          borderRadius: rem(5),
        },
        pre: {
          borderRadius: rem(8),
        },
      },
    ],
  },
  "2xl": {
    css: [
      {
        fontSize: rem(24),
        kbd: {
          borderRadius: rem(6),
        },
        pre: {
          borderRadius: rem(8),
        },
      },
    ],
  },
};
