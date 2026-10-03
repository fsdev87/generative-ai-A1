# Frontend image: build the React app with Node, then serve the static files with nginx, which
# also forwards /api to the backend service (nginx.conf). Build context = frontend/:
#   docker build -t genai-frontend frontend
# Normally started by the docker-compose.yml in the repository root.

FROM node:22-alpine AS build
WORKDIR /app
# Dependencies first, so this layer is reused while only the source code changes.
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY . .
# Type-checks (tsc) and builds the production bundle into dist/.
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1
