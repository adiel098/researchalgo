# Santa Claus Algorithm Demo

This is a full-stack web application that demonstrates the Santa Claus Problem algorithm based on the work by Bansal and Sviridenko. The application uses React for the frontend and Flask for the backend.

## Project Structure

- `frontend/`: React application
- `backend/`: Flask API server
- `fairpyx/`: Original algorithm implementation

## Getting Started

### Prerequisites

- Node.js and npm
- Python 3.8+
- pip

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```
   python app.py
   ```
   The server will run on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install the required npm packages:
   ```
   npm install
   ```

3. Start the React development server:
   ```
   npm start
   ```
   The application will open in your browser at http://localhost:3000

## Features

- **Home Page**: Overview of the Santa Claus algorithm and its purpose
- **Input Form**: Create and edit problem instances with kids, presents, and valuations
- **Random Input Generator**: Generate random problem instances for testing
- **Results Page**: View the algorithm's output, including allocations and logs
- **About Page**: Information about the developer

## Algorithm Overview

The Santa Claus Problem involves distributing presents (gifts) among kids (children), where each kid has different valuations for each present. The goal is to maximize the happiness of the least happy kid (maximin objective).

In the restricted assignment case, each present has a fixed value for all kids who can receive it, and 0 for kids who cannot receive it.

This implementation follows the O(log log m / log log log m) approximation algorithm for the restricted assignment case.

## Docker Deployment (Next Steps)

To package the application with Docker for deployment:

1. Create a `Dockerfile` for the backend:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY backend/requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY backend/ ./backend/
   COPY fairpyx/ ./fairpyx/
   
   EXPOSE 5000
   
   CMD ["python", "backend/app.py"]
   ```

2. Create a `Dockerfile` for the frontend:
   ```dockerfile
   FROM node:16-alpine as build
   
   WORKDIR /app
   
   COPY frontend/package*.json ./
   RUN npm install
   
   COPY frontend/ ./
   RUN npm run build
   
   FROM nginx:alpine
   COPY --from=build /app/build /usr/share/nginx/html
   EXPOSE 80
   CMD ["nginx", "-g", "daemon off;"]
   ```

3. Create a `docker-compose.yml` file:
   ```yaml
   version: '3'
   
   services:
     backend:
       build:
         context: .
         dockerfile: backend.Dockerfile
       ports:
         - "5000:5000"
     
     frontend:
       build:
         context: .
         dockerfile: frontend.Dockerfile
       ports:
         - "80:80"
       depends_on:
         - backend
   ```

4. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

## References

- "The Santa Claus Problem", by Bansal, Nikhil, and Maxim Sviridenko.
  Proceedings of the 38th Annual ACM Symposium on Theory of Computing, 2006
  https://dl.acm.org/doi/10.1145/1132516.1132557
