;; USDCx Escrow Contract for ApexTrader-Stacks
;; Built using patterns from clarity-lang/book and friedger/clarity-smart-contracts
;; Handles deposit, withdrawal, and trade approval for automated trading

;; ============================================
;; Constants
;; ============================================

(define-constant CONTRACT_OWNER tx-sender)

;; Error codes (following common pattern from Clarity Book)
(define-constant ERR_UNAUTHORIZED (err u100))
(define-constant ERR_INSUFFICIENT_BALANCE (err u101))
(define-constant ERR_NOT_FOUND (err u102))
(define-constant ERR_INVALID_AMOUNT (err u103))
(define-constant ERR_EXPIRED (err u104))
(define-constant ERR_ALREADY_EXECUTED (err u105))
(define-constant ERR_PAUSED (err u106))

;; ============================================
;; Data Variables
;; ============================================

(define-data-var next-trade-id uint u1)
(define-data-var paused bool false)

;; ============================================
;; Data Maps
;; ============================================

;; User balances in escrow (in microSTX)
(define-map balances principal uint)

;; Authorized bot operators
(define-map authorized-bots principal bool)

;; Approved trades awaiting execution
(define-map trades
  { id: uint }
  {
    user: principal,
    amount: uint,
    expiry: uint,
    executed: bool
  }
)

;; ============================================
;; Read-Only Functions
;; ============================================

(define-read-only (get-balance (user principal))
  (default-to u0 (map-get? balances user))
)

(define-read-only (get-trade (id uint))
  (map-get? trades { id: id })
)

(define-read-only (is-bot-authorized (bot principal))
  (default-to false (map-get? authorized-bots bot))
)

(define-read-only (get-next-trade-id)
  (var-get next-trade-id)
)

(define-read-only (is-paused)
  (var-get paused)
)

;; ============================================
;; Public Functions - User Operations
;; ============================================

;; Deposit STX into escrow
(define-public (deposit (amount uint))
  (begin
    (asserts! (not (var-get paused)) ERR_PAUSED)
    (asserts! (> amount u0) ERR_INVALID_AMOUNT)
    (try! (stx-transfer? amount tx-sender (as-contract tx-sender)))
    (map-set balances tx-sender (+ (get-balance tx-sender) amount))
    (ok amount)
  )
)

;; Withdraw STX from escrow
(define-public (withdraw (amount uint))
  (let
    (
      (caller tx-sender)
      (current-balance (get-balance tx-sender))
    )
    (asserts! (not (var-get paused)) ERR_PAUSED)
    (asserts! (> amount u0) ERR_INVALID_AMOUNT)
    (asserts! (>= current-balance amount) ERR_INSUFFICIENT_BALANCE)
    (map-set balances caller (- current-balance amount))
    (as-contract (stx-transfer? amount tx-sender caller))
  )
)

;; ============================================
;; Public Functions - Bot Operations
;; ============================================

;; Bot requests trade approval
(define-public (request-trade (user principal) (amount uint) (expiry-blocks uint))
  (let ((trade-id (var-get next-trade-id)))
    (asserts! (not (var-get paused)) ERR_PAUSED)
    (asserts! (is-bot-authorized tx-sender) ERR_UNAUTHORIZED)
    (asserts! (> amount u0) ERR_INVALID_AMOUNT)
    (asserts! (>= (get-balance user) amount) ERR_INSUFFICIENT_BALANCE)
    (map-insert trades
      { id: trade-id }
      {
        user: user,
        amount: amount,
        expiry: (+ stacks-block-height expiry-blocks),
        executed: false
      }
    )
    (var-set next-trade-id (+ trade-id u1))
    (ok trade-id)
  )
)

;; Bot executes trade and updates balance
(define-public (execute-trade (id uint) (profit int))
  (let ((trade-data (unwrap! (get-trade id) ERR_NOT_FOUND)))
    (asserts! (not (var-get paused)) ERR_PAUSED)
    (asserts! (is-bot-authorized tx-sender) ERR_UNAUTHORIZED)
    (asserts! (not (get executed trade-data)) ERR_ALREADY_EXECUTED)
    (asserts! (<= stacks-block-height (get expiry trade-data)) ERR_EXPIRED)
    (let
      (
        (user (get user trade-data))
        (current-balance (get-balance user))
        (new-balance (if (< profit 0)
                       (if (>= current-balance (to-uint (* profit -1)))
                         (- current-balance (to-uint (* profit -1)))
                         u0)
                       (+ current-balance (to-uint profit))))
      )
      (map-set balances user new-balance)
      (map-set trades { id: id }
        (merge trade-data { executed: true }))
      (ok true)
    )
  )
)

;; ============================================
;; Public Functions - Admin Operations
;; ============================================

(define-public (authorize-bot (bot principal))
  (begin
    (asserts! (is-eq tx-sender CONTRACT_OWNER) ERR_UNAUTHORIZED)
    (map-set authorized-bots bot true)
    (ok true)
  )
)

(define-public (revoke-bot (bot principal))
  (begin
    (asserts! (is-eq tx-sender CONTRACT_OWNER) ERR_UNAUTHORIZED)
    (map-delete authorized-bots bot)
    (ok true)
  )
)

(define-public (set-paused (new-paused-state bool))
  (begin
    (asserts! (is-eq tx-sender CONTRACT_OWNER) ERR_UNAUTHORIZED)
    (var-set paused new-paused-state)
    (ok true)
  )
)
