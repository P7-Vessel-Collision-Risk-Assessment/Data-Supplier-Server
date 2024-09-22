use std::time::Duration;
use tokio::{
    net::{TcpListener, TcpStream},
    signal::{self},
};

#[no_mangle]
#[tokio::main]
pub async extern "C" fn start_sse_server() {
    let address = "127.0.0.1:8080";

    let listener = TcpListener::bind(address).await.unwrap();
    println!("SSE Server Listening on http://{address}");
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(());

    let shutdown_signal = async move {
        signal::ctrl_c().await.unwrap();
        println!("Shutdown signal received");
        shutdown_tx.send(()).unwrap();
    };

    tokio::spawn(shutdown_signal);

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, addr)) => {
                        println!("Connected to client {:?}", addr);

                        tokio::spawn(async move {
                            if let Err(e) = handle_client(stream).await {
                                eprintln!("Error: {e}");
                            };
                        });
                    }
                    Err(e) => {eprintln!("Couldn't get client {e}")}
                }
            }
            _ = shutdown_rx.changed() => {

                break;
            }
        }
    }
}

async fn handle_client(mut stream: TcpStream) -> tokio::io::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let mut reader = BufReader::new(&mut stream);
    let mut buf = String::new();
    reader.read_line(&mut buf).await?;
    if buf.starts_with("GET /") {
        stream.write_all(b"HTTP/1.1 200 OK\r\n").await?;
        stream
            .write_all(b"Content-Type: text/event-stream\r\n")
            .await?;
        stream.write_all(b"Cache-Control: no-cache\r\n").await?;
        stream.write_all(b"Connection: keep-alive\r\n\r\n").await?;
        stream.flush().await?;

        let mut test = 0;

        loop {
            test += 1;

            let data = format!("data: {}\n\n", test);
            stream.write_all(data.as_bytes()).await?;
            stream.flush().await?;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    Ok(())
}
