FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps

COPY . .

ENV NEXT_TELEMETRY_DISABLED=1
RUN npx next build

EXPOSE 8080
ENV PORT=8080

CMD ["npx", "next", "start", "-p", "8080"]
