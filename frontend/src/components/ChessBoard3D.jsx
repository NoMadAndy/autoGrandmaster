import React, { useRef, useState, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'

// Chess piece representations (simplified 3D shapes)
function ChessPiece({ position, type, color, onClick }) {
  const meshRef = useRef()
  const [hovered, setHovered] = useState(false)
  
  useFrame(() => {
    if (meshRef.current && hovered) {
      meshRef.current.position.y = position[1] + 0.1
    } else if (meshRef.current) {
      meshRef.current.position.y = position[1]
    }
  })
  
  const pieceColor = color === 'w' ? '#f1f5f9' : '#1e293b'
  
  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        {/* Simplified piece shapes */}
        {type === 'p' && <cylinderGeometry args={[0.15, 0.2, 0.5, 16]} />}
        {type === 'n' && <coneGeometry args={[0.25, 0.6, 16]} />}
        {type === 'b' && <coneGeometry args={[0.2, 0.7, 16]} />}
        {type === 'r' && <boxGeometry args={[0.4, 0.6, 0.4]} />}
        {type === 'q' && <coneGeometry args={[0.3, 0.8, 8]} />}
        {type === 'k' && <cylinderGeometry args={[0.25, 0.25, 0.8, 8]} />}
        
        <meshStandardMaterial
          color={pieceColor}
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>
    </group>
  )
}

function ChessSquare({ position, color, isHighlighted, onClick }) {
  const squareColor = color === 'light' ? '#e2e8f0' : '#64748b'
  const highlightColor = '#fbbf24'
  
  return (
    <mesh
      position={position}
      rotation={[-Math.PI / 2, 0, 0]}
      onClick={onClick}
    >
      <planeGeometry args={[1, 1]} />
      <meshStandardMaterial
        color={isHighlighted ? highlightColor : squareColor}
        opacity={isHighlighted ? 0.7 : 1}
        transparent={isHighlighted}
      />
    </mesh>
  )
}

function Board({ fen, legalMoves, onMove }) {
  const [selectedSquare, setSelectedSquare] = useState(null)
  const [pieces, setPieces] = useState([])
  
  useEffect(() => {
    // Parse FEN to get piece positions
    const parseFen = (fen) => {
      const parts = fen.split(' ')
      const board = parts[0].split('/')
      const pieces = []
      
      board.forEach((row, rank) => {
        let file = 0
        for (const char of row) {
          if (isNaN(char)) {
            const color = char === char.toUpperCase() ? 'w' : 'b'
            const type = char.toLowerCase()
            const position = [file - 3.5, 0.3, (7 - rank) - 3.5]
            pieces.push({ position, type, color, square: `${String.fromCharCode(97 + file)}${8 - rank}` })
            file++
          } else {
            file += parseInt(char)
          }
        }
      })
      
      return pieces
    }
    
    setPieces(parseFen(fen))
  }, [fen])
  
  const handleSquareClick = (file, rank) => {
    const square = `${String.fromCharCode(97 + file)}${rank + 1}`
    
    if (selectedSquare) {
      const move = `${selectedSquare}${square}`
      if (legalMoves.includes(move)) {
        onMove(move)
      }
      setSelectedSquare(null)
    } else {
      const piece = pieces.find(p => p.square === square)
      if (piece && piece.color === 'w') {
        setSelectedSquare(square)
      }
    }
  }
  
  const handlePieceClick = (piece) => {
    if (piece.color === 'w') {
      setSelectedSquare(piece.square)
    }
  }
  
  const getHighlightedSquares = () => {
    if (!selectedSquare) return []
    return legalMoves
      .filter(move => move.startsWith(selectedSquare))
      .map(move => move.substring(2, 4))
  }
  
  const highlightedSquares = getHighlightedSquares()
  
  return (
    <group>
      {/* Board squares */}
      {Array.from({ length: 8 }).map((_, rank) =>
        Array.from({ length: 8 }).map((_, file) => {
          const square = `${String.fromCharCode(97 + file)}${rank + 1}`
          const isLight = (file + rank) % 2 === 0
          const isHighlighted = highlightedSquares.includes(square)
          
          return (
            <ChessSquare
              key={`${file}-${rank}`}
              position={[file - 3.5, 0, rank - 3.5]}
              color={isLight ? 'light' : 'dark'}
              isHighlighted={isHighlighted}
              onClick={() => handleSquareClick(file, rank)}
            />
          )
        })
      )}
      
      {/* Pieces */}
      {pieces.map((piece, index) => (
        <ChessPiece
          key={index}
          position={piece.position}
          type={piece.type}
          color={piece.color}
          onClick={() => handlePieceClick(piece)}
        />
      ))}
      
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <pointLight position={[-10, 10, -5]} intensity={0.5} />
    </group>
  )
}

function ChessBoard3D({ fen, legalMoves, onMove, isPlayerTurn }) {
  return (
    <div style={{ width: '100%', height: '600px' }}>
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0, 8, 8]} fov={50} />
        <OrbitControls
          enablePan={false}
          minDistance={5}
          maxDistance={15}
          maxPolarAngle={Math.PI / 2.5}
        />
        
        <Board fen={fen} legalMoves={legalMoves} onMove={onMove} />
        
        {/* Base platform */}
        <mesh position={[0, -0.1, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[10, 10]} />
          <meshStandardMaterial color="#0f172a" />
        </mesh>
      </Canvas>
    </div>
  )
}

export default ChessBoard3D
