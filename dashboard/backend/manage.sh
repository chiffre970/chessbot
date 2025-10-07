#!/bin/bash
# Dashboard Backend Management Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000

case "$1" in
    start)
        echo "Starting dashboard backend..."
        cd "$SCRIPT_DIR"
        source venv/bin/activate
        python -m app.main --host 0.0.0.0 --port $PORT &
        echo "Server started in background"
        sleep 2
        if lsof -ti:$PORT > /dev/null 2>&1; then
            echo "✓ Server is running on http://localhost:$PORT"
            echo "✓ API docs: http://localhost:$PORT/docs"
            echo "✓ PID: $(lsof -ti:$PORT)"
        else
            echo "✗ Failed to start server"
            exit 1
        fi
        ;;
    
    stop)
        echo "Stopping dashboard backend..."
        if lsof -ti:$PORT > /dev/null 2>&1; then
            PID=$(lsof -ti:$PORT)
            kill $PID
            echo "✓ Server stopped (PID: $PID)"
        else
            echo "✓ Server is not running"
        fi
        ;;
    
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    
    status)
        if lsof -ti:$PORT > /dev/null 2>&1; then
            PID=$(lsof -ti:$PORT)
            echo "✓ Server is running"
            echo "  - PID: $PID"
            echo "  - URL: http://localhost:$PORT"
            echo "  - API docs: http://localhost:$PORT/docs"
            echo ""
            # Test health endpoint
            if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
                echo "✓ Health check: OK"
            else
                echo "✗ Health check: FAILED"
            fi
        else
            echo "✗ Server is not running"
            exit 1
        fi
        ;;
    
    logs)
        echo "Showing recent logs (Ctrl+C to stop)..."
        cd "$SCRIPT_DIR"
        if [ -f "server.log" ]; then
            tail -f server.log
        else
            echo "No log file found. Server might be running in foreground."
        fi
        ;;
    
    *)
        echo "Dashboard Backend Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the server in background"
        echo "  stop    - Stop the server"
        echo "  restart - Restart the server"
        echo "  status  - Check server status"
        echo "  logs    - Show server logs (if available)"
        echo ""
        exit 1
        ;;
esac

